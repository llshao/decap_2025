* RC 低通滤波器 + AC 灵敏度分析（可复现测试用例）
* Vin -> R1 -> Out -> C1 -> GND

Vin in 0 AC 1
R1  in out 1k
C1  out 0 1u

* 常规 AC 扫频（10 Hz ~ 1 MHz，按 decade 每 decade 20 点）
.ac dec 20 10 1e6

.control
  * --- 1) 常规 AC 频响导出为 CSV ---
  set wr_singlescale
  set wr_vecnames
  set filetype=ascii
  run
  * 导出：频率、线性幅值、分贝、相位（度）
  wrdata rc_ac.csv frequency v(out) vdb(out) vp(out)

  * --- 2) 进行 AC 灵敏度分析 ---
  * 用控制命令 sens 执行 AC 灵敏度（对 v(out)）
  sens v(out) ac dec 20 10 1e6
  * sens 会生成一个新的“plot”，包含各器件参数对输出的复数灵敏度向量

  * --- 3) 把灵敏度结果完整保存（ASCII raw，易于后处理） ---
  set filetype=ascii
  * 写出所有 plot（包含 sens 的那个）到 ASCII raw
  write rc_all_plots.raw

  * 也可以只写出当前（灵敏度）plot：
  * setplot sens_ac1
  * write rc_sens_only.raw

  quit
.endc

.end
