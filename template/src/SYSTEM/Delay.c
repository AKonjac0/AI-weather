#include "stm32f10x.h"

void RCC_Configuration(void) {
  RCC_DeInit();
  RCC_HSEConfig(RCC_HSE_ON);
  ErrorStatus HSEStartUpStatus = RCC_WaitForHSEStartUp();
  if (HSEStartUpStatus == SUCCESS) {
    RCC_HCLKConfig(RCC_SYSCLK_Div1);
    RCC_PCLK2Config(RCC_HCLK_Div1);
    RCC_PCLK1Config(RCC_HCLK_Div2);
    FLASH_SetLatency(FLASH_Latency_2);
    FLASH_PrefetchBufferCmd(FLASH_PrefetchBuffer_Enable);
    RCC_PLLConfig(RCC_PLLSource_HSE_Div1, RCC_PLLMul_9);
    RCC_PLLCmd(ENABLE);
    while (RCC_GetFlagStatus(RCC_FLAG_PLLRDY) == RESET)
      ;
    RCC_SYSCLKConfig(RCC_SYSCLKSource_PLLCLK);
    while (RCC_GetSYSCLKSource() != 0x08)
      ;
  }
}

/**
 * @brief  微秒级延时
 * @param  xus 延时时长，范围：0~233015
 * @retval 无
 */
void Delay_us(uint32_t xus) {
  SysTick->LOAD = 72 * xus;   // 设置定时器重装值
  SysTick->VAL = 0x00;        // 清空当前计数值
  SysTick->CTRL = 0x00000005; // 设置时钟源为HCLK，启动定时器
  while (!(SysTick->CTRL & 0x00010000))
    ;                         // 等待计数到0
  SysTick->CTRL = 0x00000004; // 关闭定时器
}

/**
 * @brief  毫秒级延时
 * @param  xms 延时时长，范围：0~4294967295
 * @retval 无
 */
void Delay_ms(uint32_t xms) {
  while (xms--) {
    Delay_us(1000);
  }
}

/**
 * @brief  秒级延时
 * @param  xs 延时时长，范围：0~4294967295
 * @retval 无
 */
void Delay_s(uint32_t xs) {
  while (xs--) {
    Delay_ms(1000);
  }
}
