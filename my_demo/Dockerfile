FROM ubuntu:18.04
# sed -i 's/aaa/bbb/g' 用bbb替换aaa
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && apt update && apt install -y libeigen3-dev cmake && apt clean
COPY entrypoint.sh /entrypoint.sh
# WORKDIR指定工作目录
# /slambook2是绝对路径
WORKDIR /slambook2
# 为镜像指定容器启动后的命令
# ENTRYPOINT相当于CMD，每个Dockerfile只有一个ENTRYPOINT
# ENTRYPOINT不会被docker启动参数覆盖
ENTRYPOINT [ "/entrypoint.sh" ]