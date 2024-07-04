#include "stm32f10x.h"
#include "Delay.h"
#include "stm32f10x_rcc.h"
#include "stm32f10x_gpio.h"
#include "Serial.h"
#include "LCD1602.h"
#include "dht11.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
double ave[10]={0,17.8917449  ,18.37022743, 17.43894925, 70.12909429, 72.43377444, 67.81805314};
double st[10]={0,9.1506803 ,  9.24498003,  9.06754299, 19.23407723, 18.60495473, 19.86942873};
double conv1_weight[40]={0,-0.9344, -0.7177,0.3404,  1.0682,-0.3441, -0.7795,
-0.1951, -0.5403,-0.1641, -0.2621,-1.2663, -0.8505,
0.3434, -0.8203,0.3385, -0.2824,0.9545,  0.9977,-0.5027, -0.6400,0.0880, -0.2460,
0.1680, -0.1745,0.3735, -0.8991,0.4298,  1.0205,0.0766, -0.2049,0.0433,  0.0471};
double conv1_bias[20]={0,-0.3971, -0.1607,  1.5290,  0.0930,  0.8519, -0.3865,  0.5710,  0.5718,
        -0.4650,  0.2558, -0.7117,  1.2057,  0.6273, -0.5318,  0.9053,  1.0075};
double fc1_weight[40]={0, 0.3256, -0.0064,  0.0618,  0.4257,  0.0459, -0.4259,  0.1012,  0.5800,
         -0.1315, -0.2526,  0.1875, -0.2139, -0.1266, -0.7136, -0.3686, -0.3266,
         -0.1834,  0.8236, -0.1437, -0.1398, -0.0214,  0.0068, -0.3845, -0.2273,
         -0.1736, -0.4593, -0.0957,  0.7517, -0.3786, -0.4038, -0.1912, -0.1260};
double fc1_bias=-0.2754;

double input[10];
double conv[20][6];
double pool[20][3];
double view[40];
double max(double a,double b){
  return a>b?a:b;
}
double sigmoid(double x){
	return 1.0/(1.0+exp(-x));
}

double calc(){
  for(int i=1;i<=16;++i){
		double a=conv1_weight[(i-1)*2+1],b=conv1_weight[(i)*2];
		for(int j=1;j+1<=6;++j){
			conv[i][j]=max(0.0,input[j]*a+input[j+1]*b+conv1_bias[i]);	
		}
	}
	for(int i=1;i<=16;++i){
		for(int j=1;j<=2;++j) pool[i][j]=max(conv[i][(j-1)*2+1],conv[i][(j)*2]);
		
	}
	for(int i=1;i<=16;++i){
		for(int j=1;j<=2;++j) view[(i-1)*2+j]=pool[i][j];
	}
	double pred=0;
	for(int i=1;i<=32;++i){
		pred+=view[i]*fc1_weight[i];
	}
	pred+=fc1_bias;
	return sigmoid(pred);
}
void LED_Init(){
  GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Pin	= GPIO_Pin_1;
	GPIO_InitStructure.GPIO_Speed	= GPIO_Speed_50MHz;//选择工作频率
	GPIO_InitStructure.GPIO_Mode	= GPIO_Mode_Out_PP;//设置工作模式
	GPIO_Init( GPIOB, &GPIO_InitStructure );
}
int main(void){

  uint16_t _temperature, _humidity;
  double temperature, humidity, oldt, oldh;
  
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE); // 不要忘记使能DHT所在GPIO口的时钟 !!!
  DHT11_Init();

  
  

  LED_Init();
  RCC_Configuration();
  // 使能APB2外设时钟
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);

  Serial_Init();
  LCD1602_Init();

  
  // LED灯闪烁
  while (1)
  {
    /*
    GPIO_WriteBit(GPIOA, GPIO_Pin_0, Bit_RESET);
    Delay_ms(500);
    GPIO_WriteBit(GPIOA, GPIO_Pin_0, Bit_SET);
    Delay_ms(500);
    */
   Delay_ms(1000);
    
    
    DHT11_Read_Data(&_temperature, &_humidity);
    temperature = (_temperature >> 8) + (_temperature & 0xff)*0.1; // 单位℃
    humidity = (_humidity >> 8) + (_humidity & 0xff)*0.1; // 单位%

    if(oldt!= temperature || oldh!=humidity){
      Serial_Printf("temperature:%.6lf\n",temperature);
      Serial_Printf("humidity:%.6lf\n",humidity);
      //LCD1602_ClearScreen();
      char temp[20],hum[20];
      sprintf(temp, "%.2lf", temperature);
      sprintf(hum, "%.2lf", humidity);
      LCD1602_Show_Str(5, 0, "C");
      //LCD1602_Show_Str(12, 0, "");
      LCD1602_Show_Str(0, 0, temp);
      LCD1602_Show_Str(7, 0, hum);
      LCD1602_Show_Str(0, 1, "Weather:");
      
      input[1]=input[2]=input[3]=temperature;
      input[4]=input[5]=input[6]=humidity;
      for(int i=1;i<=6;++i){
		    input[i]=(input[i]-ave[i])/st[i];
	    }
      double res = calc();
      if(res>0.5){
        LCD1602_Show_Str(8, 1, "Rain "),GPIO_WriteBit(GPIOB, GPIO_Pin_1, Bit_RESET);
      }else LCD1602_Show_Str(8, 1, "Clear"),GPIO_WriteBit(GPIOB, GPIO_Pin_1, Bit_SET);
    }
    oldt = temperature;
    oldh = humidity;
    
  }
}
