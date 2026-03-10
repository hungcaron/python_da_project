import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self, df, chart_dir=None):
        """
        Khởi tạo DataCleaner
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame cần làm sạch
        chart_dir : str, optional
            Thư mục lưu biểu đồ
        """
        self.df = df.copy()
        self.chart_dir = chart_dir
        self.cleaning_summary = {}
        self.df_cleaned = None
        
    def check_missing_values(self):
        """Kiểm tra và xử lý giá trị thiếu"""
        missing_info = self.df.isnull().sum()
        missing_percent = (missing_info / len(self.df)) * 100
        
        missing_report = pd.DataFrame({
            'missing_count': missing_info,
            'missing_percent': missing_percent
        }).sort_values('missing_count', ascending=False)
        
        # Lưu thông tin missing
        missing_data = missing_report[missing_report['missing_count'] > 0]
        
        if len(missing_data) == 0:
            self.cleaning_summary['missing_values'] = {
                'status': 'No missing values found',
                'missing_data': None,
                'action_taken': 'None'
            }
        else:
            # Xử lý missing values (ví dụ: fill với mode hoặc median)
            for col in missing_data.index:
                if self.df[col].dtype == 'object':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                else:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
            
            self.cleaning_summary['missing_values'] = {
                'status': f'Found and handled {len(missing_data)} columns with missing values',
                'missing_data': missing_data.to_dict(),
                'action_taken': 'Filled with mode (categorical) or median (numerical)'
            }
        
        return self.df
    
    def check_duplicates(self):
        """Kiểm tra và xử lý bản ghi trùng lặp"""
        duplicate_count = self.df.duplicated().sum()
        
        if duplicate_count > 0:
            original_shape = self.df.shape
            self.df = self.df.drop_duplicates()
            removed_count = original_shape[0] - self.df.shape[0]
            
            self.cleaning_summary['duplicates'] = {
                'status': f'Found and removed {removed_count} duplicate records',
                'duplicate_count': int(duplicate_count),
                'action_taken': 'Removed duplicates'
            }
        else:
            self.cleaning_summary['duplicates'] = {
                'status': 'No duplicates found',
                'duplicate_count': 0,
                'action_taken': 'None'
            }
        
        return self.df
    
    def check_data_types(self):
        """Kiểm tra và chuẩn hóa kiểu dữ liệu"""
        type_changes = []
        original_dtypes = self.df.dtypes.to_dict()
        
        # Chuyển đổi cột datetime nếu cần
        datetime_columns = ['Order Date', 'Ship Date']
        for col in datetime_columns:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    type_changes.append(f"{col}: converted to datetime")
                except:
                    type_changes.append(f"{col}: failed to convert to datetime")
        
        # Kiểm tra cột numeric
        numeric_columns = ['Sales', 'Quantity', 'Discount', 'Profit']
        for col in numeric_columns:
            if col in self.df.columns:
                if self.df[col].dtype == 'object':
                    try:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                        type_changes.append(f"{col}: converted to numeric")
                    except:
                        type_changes.append(f"{col}: failed to convert to numeric")
        
        new_dtypes = self.df.dtypes.to_dict()
        
        self.cleaning_summary['data_types'] = {
            'status': f'Checked {len(datetime_columns + numeric_columns)} columns for type consistency',
            'type_changes': type_changes,
            'original_dtypes': original_dtypes,
            'new_dtypes': new_dtypes
        }
        
        return self.df
    
    def handle_outliers(self):
        """Phát hiện và xử lý outliers cho các cột số"""
        outliers_info = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['Sales', 'Profit', 'Quantity']:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    outliers_info[col] = {
                        'outlier_count': int(outlier_count),
                        'outlier_percent': float((outlier_count / len(self.df)) * 100),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
                    
                    # Capping outliers (thay vì xóa)
                    self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, 
                                          np.where(self.df[col] < lower_bound, lower_bound, self.df[col]))
        
        if outliers_info:
            self.cleaning_summary['outliers'] = {
                'status': f'Found and capped outliers in {len(outliers_info)} numeric columns',
                'outliers_details': outliers_info,
                'action_taken': 'Capped using IQR method (1.5*IQR)'
            }
        else:
            self.cleaning_summary['outliers'] = {
                'status': 'No significant outliers found or handled',
                'outliers_details': None,
                'action_taken': 'None'
            }
        
        return self.df
    
    def standardize_categories(self):
        """Chuẩn hóa các giá trị categorical"""
        category_issues = []
        
        # Không đổi tên cột, chỉ chuẩn hóa giá trị
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in self.df.columns:
                # Loại bỏ khoảng trắng thừa
                original_unique = self.df[col].nunique()
                self.df[col] = self.df[col].astype(str).str.strip()
                new_unique = self.df[col].nunique()
                
                if original_unique != new_unique:
                    category_issues.append(f"{col}: reduced from {original_unique} to {new_unique} unique values")
        
        self.cleaning_summary['categories'] = {
            'status': f'Standardized {len(categorical_cols)} categorical columns',
            'category_issues': category_issues,
            'action_taken': 'Trimmed whitespace and standardized values'
        }
        
        return self.df
    
    def create_derived_features(self):
        """Tạo các đặc trưng mới từ dữ liệu hiện có"""
        derived_features = []
        
        # Tạo cột Year, Month từ Order Date
        if 'Order Date' in self.df.columns:
            self.df['Order_Year'] = self.df['Order Date'].dt.year
            self.df['Order_Month'] = self.df['Order Date'].dt.month
            self.df['Order_Quarter'] = self.df['Order Date'].dt.quarter
            self.df['Order_Day'] = self.df['Order Date'].dt.day
            derived_features.extend(['Order_Year', 'Order_Month', 'Order_Quarter', 'Order_Day'])
        
        # Tạo cột Shipping Days
        if all(col in self.df.columns for col in ['Order Date', 'Ship Date']):
            self.df['Shipping_Days'] = (self.df['Ship Date'] - self.df['Order Date']).dt.days
            derived_features.append('Shipping_Days')
        
        # Tạo cột Total Cost (Sales - Profit)
        if all(col in self.df.columns for col in ['Sales', 'Profit']):
            self.df['Total_Cost'] = self.df['Sales'] - self.df['Profit']
            derived_features.append('Total_Cost')
        
        # Tạo cột Profit Margin
        if all(col in self.df.columns for col in ['Sales', 'Profit']):
            self.df['Profit_Margin'] = np.where(
                self.df['Sales'] != 0, 
                (self.df['Profit'] / self.df['Sales']) * 100, 
                0
            )
            derived_features.append('Profit_Margin')
        
        self.cleaning_summary['derived_features'] = {
            'status': f'Created {len(derived_features)} new derived features',
            'features_created': derived_features,
            'description': 'Added temporal features, shipping metrics, and financial ratios'
        }
        
        return self.df
    
    def save_clean_data(self, filepath):
        """Lưu dữ liệu đã làm sạch"""
        if self.df is not None:
            self.df.to_csv(filepath, index=False)
            self.cleaning_summary['save_status'] = {
                'status': 'Success',
                'filepath': filepath,
                'rows': len(self.df),
                'columns': len(self.df.columns)
            }
        else:
            self.cleaning_summary['save_status'] = {
                'status': 'Failed',
                'error': 'No cleaned data available'
            }
    
    def run_all(self):
        """
        Chạy toàn bộ quy trình làm sạch dữ liệu
        
        Returns:
        --------
        dict: Dictionary chứa tất cả kết quả làm sạch
        """
        print("Starting data cleaning process...")
        
        # Chạy các bước làm sạch
        steps = [
            ("Checking missing values", self.check_missing_values),
            ("Checking duplicates", self.check_duplicates),
            ("Checking data types", self.check_data_types),
            ("Handling outliers", self.handle_outliers),
            ("Standardizing categories", self.standardize_categories),
            ("Creating derived features", self.create_derived_features)
        ]
        
        for step_name, step_func in steps:
            print(f"  - {step_name}")
            try:
                step_func()
            except Exception as e:
                print(f"    Error in {step_name}: {str(e)}")
        
        # Tạo báo cáo tổng quan
        self.df_cleaned = self.df
        
        cleaning_results = {
            'summary': {
                'original_shape': None,  # Có thể lưu shape gốc nếu cần
                'cleaned_shape': self.df_cleaned.shape,
                'rows_removed': 0,  # Tính toán nếu có
                'columns_added': len([col for col in self.df_cleaned.columns 
                                     if col not in self.df.columns]) if hasattr(self, 'original_df') else 0,
                'cleaning_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'cleaning_steps': self.cleaning_summary,
            'sample_data': self.df_cleaned.head(10).to_dict('records'),
            'column_info': {
                'total_columns': len(self.df_cleaned.columns),
                'numeric_columns': list(self.df_cleaned.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(self.df_cleaned.select_dtypes(include=['object']).columns),
                'datetime_columns': list(self.df_cleaned.select_dtypes(include=['datetime64']).columns)
            },
            'data_quality_metrics': {
                'missing_values_percent': float(self.df_cleaned.isnull().sum().sum() / (self.df_cleaned.shape[0] * self.df_cleaned.shape[1]) * 100),
                'duplicate_rows_percent': 0,  # Tính toán nếu cần
                'memory_usage_mb': float(self.df_cleaned.memory_usage(deep=True).sum() / 1024**2)
            }
        }
        
        print(f"✓ Data cleaning completed. Cleaned shape: {self.df_cleaned.shape}")
        
        return cleaning_results
    
    def get_cleaned_data(self):
        """Trả về DataFrame đã được làm sạch"""
        return self.df_cleaned