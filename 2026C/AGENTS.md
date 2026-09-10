
该项目为2026年全国大学生数学建模大赛C题的代码仓库。

C题文档：Mathematical-Modelling/2026C/CUMCM 2026 C题/C题.pdf

python运行环境限定为：conda虚拟环境2026C

根目录下除去题目，有4个文件夹：scripts_figures、scripts_tables、outputs、docs：
- scripts_figures 和 scripts_tables 中的脚本代码输出都放在 outputs 里面。
- outputs 文件夹分为 tables、figures、results 三个子文件夹，分别存放图、表、C题附件5的结果表。
- docs文件夹存放相关文档：symbols.md记录所有出现的符号。

命名：
- 脚本代码文件统一命名格式为：1_association.py。其中，前面的“1”表示这是问题一，后面的“association”表示这份脚本代码做的工作。
- 生成的图表统一命名格式为：1_独立性检验.csv，1_独立性检验.csv.png。其中，前面的“1”表示这是问题一，后面的“独立性检验”表示这张图或表所表示的内容。
- 若命名为“0”，表示这是预处理阶段。

根目录的文件：
- 代码会用到的python包文件全部都在根目录下的requirements.txt里。
- 根目录中的README.md为本项目的完整介绍。