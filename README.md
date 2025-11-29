# AI-weather
simple weather forecast 

根据温湿度预测天气是否下雨

部署在stm32上

模型很简单，效果也不咋地，湿度90%以上报告下雨

使用神经网络训练模型，将参数写到C语言代码，再下载到stm32上

stm32接了温湿度计，LED灯和LCD1602屏幕

platformIO使用方法略。

```./template/src/main.c```是源代码, 其他自用库放在```./template/src/SYSTEM```里。

编译C代码记得在```./template/platformio.ioi```的```build_flags```里加上```-Wl,-u_printf_float```，允许浮点数运算


