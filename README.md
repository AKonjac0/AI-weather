# AI-weather
easy weather forecast 

根据温湿度预测天气是否下雨

部署在stm32上

模型很简单，效果也不咋地，湿度90%以上报告下雨

使用卷积神经网络(~~线性预测为什么用卷积神经网络，这很~~)训练模型，将参数写到C语言代码，再下载到stm32上

stm32接了温湿度计，LED灯和LCD1602屏幕

```./template/src/main.c```是源代码, 其他自用库放在```./template/src/SYSTEM```里。

~~评价：挺好玩的~~

