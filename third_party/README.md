# 随仓库分发的第三方依赖

本目录包含 MLVC 编译所需、可随源码分发的第三方代码：

- `tomlplusplus/include/`：TOML 配置解析头文件；
- `msrtc_rans/`：MLVC 官方 rANS C++ 实现及其公共/私有头文件。

顶层 CMake 会固定从本目录引用这些文件，不依赖仓库外的同名目录。
Ascend CANN、DVPP 和 OpenCV 属于设备系统或运行环境依赖，不在本目录
重复打包。
