#ifndef __LCD1602_H
#define __LCD1602_H 
#define u8 uint8_t
#include <stdio.h>
/***************************根据自己的硬件引脚做修改*****************************/
#define LCD_RS_Set()	GPIO_SetBits( GPIOB, GPIO_Pin_12 )//1602的数据/指令选择控制线
#define LCD_RS_Clr()	GPIO_ResetBits( GPIOB, GPIO_Pin_12 )

#define LCD_RW_Set()	GPIO_SetBits( GPIOB, GPIO_Pin_13 )//1602的读写控制线
#define LCD_RW_Clr()	GPIO_ResetBits( GPIOB, GPIO_Pin_13 )

#define LCD_EN_Set()	GPIO_SetBits( GPIOB, GPIO_Pin_14 )//1602的使能控制线
#define LCD_EN_Clr()	GPIO_ResetBits( GPIOB, GPIO_Pin_14 )

#define DATAOUT( x ) GPIO_Write( GPIOA, x )	//1602的8条数据控制线

void GPIO_Configuration();

void LCD1602_Init();

void LCD1602_Wait_Ready();

void LCD1602_Write_Cmd( u8 cmd );

void LCD1602_Write_Dat( u8 data );

void LCD1602_ClearScreen();

void LCD1602_Set_Cursor( u8 x, u8 y );

void LCD1602_Show_Str( u8 x, u8 y, u8 *str );

#endif
