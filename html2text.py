import os
from bs4 import BeautifulSoup
from config_fn import html_input_dir, html_output_dir

def html2text(html_input_dir, html_output_dir):
    # 确保输出目录存在
    os.makedirs(html_output_dir, exist_ok=True) 

    # 遍历输入目录中的所有 HTML 文件
    for filename in os.listdir(html_input_dir):
        if filename.endswith('.html'):
            input_file_path = os.path.join(html_input_dir, filename)
            output_file_path = os.path.join(html_output_dir, filename.replace('.html', '.txt'))
            
            # 打开并读取 HTML 文件内容
            with open(input_file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            # 使用 BeautifulSoup 解析 HTML 内容
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取文本内容
            text_content = soup.get_text()
            
            # 将提取的文本保存到输出目录中的相应文件中，不保存换行和空格
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(text_content.replace(' ', '').replace('\n', ''))

    print("文本提取完成并保存到 'dataset/train/html_text' 目录中。")

if __name__ == '__main__':
    html2text(html_input_dir, html_output_dir)