#!/bin/bash
 CHAPTER=$1
 if [ "x${CHAPTER}" == "xbash" ];then
    exec /bin/bash
    exit $?
 fi
 cd /slambook2/${CHAPTER}
 cmake .
 make all
 # egrep：在文件内查找指定字符串
 # awk：-F指定分隔符 打印
 EXECABLE=$(egrep '^add_executable' CMakeLists.txt|head -1|awk -F'(' '{print $2}' |awk '{print $1}')
 /slambook2/${CHAPTER}/${EXECABLE}

