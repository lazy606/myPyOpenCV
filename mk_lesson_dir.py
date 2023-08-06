#! python3
# mk_lesson_dir.py - 创建每章学习文件夹

import os

if __name__ == '__main__':

    # 1.检查当前目录学到了第几课
    lesson_nums = []
    for file in os.listdir():
        if os.path.isdir(file) and file.startswith("lesson_"):
            lesson_nums.append(int(file.split(sep="_")[1]))
    num = 1
    while num in lesson_nums:
        num += 1

    # 2.根据课号创建文件夹子和响应的文件，返回程序执行结果
    os.mkdir(f"lesson_{num}")
    print(f"目录:lesson_{num}创建成功")
    with open(os.path.join(f"lesson_{num}", "example.py"), "w+"):
        pass
    print("练习文件:example.py创建成功")
