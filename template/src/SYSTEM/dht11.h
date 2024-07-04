#ifndef __DHT11_H__
#define __DHT11_H__
#include "stm32f10x.h"

#define DHT11_DQ_OUT_HIGH GPIO_WriteBit(GPIOB, GPIO_Pin_15, Bit_SET)
#define DHT11_DQ_OUT_LOW GPIO_WriteBit(GPIOB, GPIO_Pin_15, Bit_RESET)

#define DHT11_DQ_IN GPIO_ReadInputDataBit(GPIOB, GPIO_Pin_15)

uint8_t DHT11_Init(void);                                // 初始化DHT11
uint8_t DHT11_Read_Data(uint16_t *temp, uint16_t *humi); // 读取温湿度
uint8_t DHT11_Read_Byte(void);                           // 读出一个字节
uint8_t DHT11_Read_Bit(void);                            // 读出一个位
uint8_t DHT11_Check(void); // 检测是否存在DHT11
void DHT11_Rst(void);      // 复位DHT11

#endif
