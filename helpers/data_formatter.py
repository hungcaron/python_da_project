# Ví dụ về các hàm trong data_formatter.py

def format_integer(number):
    # Định dạng số nguyên với dấu phẩy phân cách hàng nghìn.
    return f"{number:,.0f}"

def format_decimal(number, decimals=2):
    # Định dạng số thập phân với số chữ số sau dấu phẩy và dấu phẩy phân cách hàng nghìn.
    return f"{number:,.{decimals}f}"

def format_percentage(number):
    # Định dạng số thành phần trăm.
    return f"{number:.2%}"

def print_as_numbered_list(data_list):
    # In danh sách dữ liệu ra dạng list có đánh số thứ tự.
    report_list = ""
    for i, item in enumerate(data_list, 1):
        report_list += f"{i}. {item}\n"
    return report_list
def format_natural_list(data_list):
    # Định dạng danh sách các chuỗi thành một chuỗi văn bản tự nhiên, 
    # sử dụng dấu phẩy và từ 'và'.
    if not data_list:
        return ""
    
    num_items = len(data_list)
    
    # Trường hợp 1: Chỉ 1 phần tử
    if num_items == 1:
        return data_list[0]
        
    # Trường hợp 2: Chỉ 2 phần tử
    elif num_items == 2:
        return f"{data_list[0]} và {data_list[1]}"
        
    # Trường hợp 3: >= 3 phần tử
    else:
        # Lấy tất cả phần tử trừ phần tử cuối cùng
        initial_part = ", ".join(data_list[:-1])
        # Nối phần đầu với từ 'và' và phần tử cuối cùng
        return f"{initial_part}, và {data_list[-1]}"
#
def list_to_text(items):
    items = list(items)  # đảm bảo là list

    if len(items) == 0:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} và {items[1]}"

    # 3 phần tử trở lên
    return ", ".join(items[:-1]) + f" và {items[-1]}"
