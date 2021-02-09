IMAGE ?= slambook2:1.0

build:
	docker build -t ${IMAGE} .

run:
	docker run --rm -v /www/hxy/docker_share_file/slambook2-master:/slambook2 ${IMAGE} ch2