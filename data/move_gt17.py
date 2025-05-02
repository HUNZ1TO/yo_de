import os
import shutil
import traceback

"""
MOT17数据集，移动到指定文件
"""

gt_dir_path = "MOT17\\train"
des_dir_path = "gt\\mot_challenge\\MOT17-train" #移动到指定目录
seq_info = "seqinfo.ini"
seq_txt = "gt\\mot_challenge\\seqmaps\\MOT17-train.txt"

def copyfile(srcfile, dstpath, replace=True):
    """复制文件到指定文件夹
    @param srcfile: 原文件绝对路径
    @param dstpath: 目标文件夹
    @param replace: 如果目标文件夹已存在同名文件，直接覆盖
    """
    try:
        if not os.path.isfile(srcfile):
            print("%s not exist!" % (srcfile))
        else:
            fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
            suffix = os.path.splitext(srcfile)[-1]
            # print(fpath, fname, suffix)
            if not os.path.exists(dstpath):
                os.makedirs(dstpath)  # 创建路径
            if replace:
                dstfile = os.path.join(dstpath, fname)
                shutil.copy(srcfile, dstfile)  # 复制文件
                print("copy %s -> %s" % (srcfile, dstfile))
            else:
                i = 1
                while True:
                    add = ' (%s)' % str(i) if i != 1 else ''
                    dstfile = os.path.join(dstpath, fname.replace(suffix, add + suffix))
                    if os.path.exists(dstfile) and i <= 10:
                        i += 1
                    else:
                        shutil.copy(srcfile, dstfile)  # 复制文件
                        print("copy %s -> %s" % (srcfile, dstfile))
                        break
            return dstfile

    except Exception as e:
        print('文件复制失败', srcfile)
        traceback.print_exc()


if __name__ == '__main__':
    if os.path.exists(seq_txt):
        os.remove(seq_txt)
    with open(seq_txt,"a+") as f:
        f.write("name"+"\n")
    ffs = [file for file in os.listdir(gt_dir_path)]
    ffs.sort()
    all_file = []
    for ff1 in ffs:
        """
        每个检测器的数据，只记录一次

        """
        temp = ff1.split("-")
        ff = ""
        for i in range(len(temp)-1):
            ff += temp[i]
            if i != len(temp)-2:
                ff +="-"
        if ff not in all_file:
            all_file.append(ff)
        else:
            continue
        
        with open(seq_txt, "a+") as f:
            f.write(ff + "\n")
        dir_name = ff1 # 原始名
        dir_path = os.path.join(des_dir_path,ff) #移动到目的文件目录名
        if os.path.exists(dir_path) == False:
            os.mkdir(dir_path)

        seq_path = os.path.join(gt_dir_path,dir_name,seq_info) #原始的ini文件路径
        copyfile(seq_path, dir_path,True) #复制ini文件


        gt_path = os.path.join(dir_path,"gt") #新建gt文件夹
        if os.path.exists(gt_path) == False:
            os.mkdir(gt_path)

        gt_txt_path = os.path.join(gt_dir_path,dir_name,'gt/gt.txt') #原始的gt文件路径
        copyfile(gt_txt_path,gt_path,True)  # 复制ini文件


