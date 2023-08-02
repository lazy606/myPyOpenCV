#! python3
# mk_lesson_dir.py - 创建每章学习文件夹

import os

if __name__ == '__main__':

    # 1.检查当前目录学到了第几课
    num = 0
    for file in os.listdir():
        if os.path.isdir(file) and file.startswith("lesson_"):
            num += 1
    lesson = num + 1

    # 2.根据课号创建文件夹子和响应的文件，返回程序执行结果
    os.mkdir(f"lesson_{lesson}")
    print(f"目录:lesson_{lesson}创建成功")

    with open(os.path.join(f"lesson_{lesson}", "example.py"), "w+"):
        pass
    print("练习文件:example.py创建成功")