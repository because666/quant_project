/**
 * 因子探索页面
 * 展示因子分布散点图、收益相关性、因子IC值、因子说明
 * 数据来源：/data/factor_data.json（从后端生成的真实因子分析数据）
 *
 * @author 量化策略系统
 * @version 2.0
 */
import { useState, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import { BarChart } from '../components/Charts'
import { ScrollReveal } from '../components/motion'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'

/** 因子收益相关性数据项 */
interface ReturnCorrelationItem {
  /** 因子名称 */
  factor: string
  /** 与未来收益的相关系数 */
  correlation: number
  /** 信息系数（IC值） */
  ic: number
}

/** 散点数据点 */
interface ScatterPoint {
  /** 因子值（x轴） */
  x: number
  /** 收益率（y轴） */
  y: number
}

/** 因子数据JSON结构 */
interface FactorData {
  /** 所有因子名称列表 */
  factors: string[]
  /** 因子间相关系数矩阵（当前为占位，仅1x1） */
  correlationMatrix: number[][]
  /** 各因子与未来收益的相关性和IC值 */
  returnCorrelation: ReturnCorrelationItem[]
  /** 全局散点数据（备用） */
  scatterData: ScatterPoint[]
  /** 按因子分组的散点数据 */
  scatterDataByFactor: Record<string, ScatterPoint[]>
}

/** 因子分类定义 */
interface FactorCategory {
  /** 分类名称 */
  name: string
  /** 分类描述 */
  description: string
  /** 该分类下的因子列表 */
  factors: string[]
}

/** 因子分类配置：与模型52个实际因子一致 */
const FACTOR_CATEGORIES: FactorCategory[] = [
  {
    name: '动量因子',
    description: '过去N日收益率，反映价格趋势与动量效应',
    factors: [
      'mom_1m',
      'mom_2m',
      'mom_3m',
      'mom_4m',
      'mom_6m',
      'mom_12m',
      'mom_accel_1m',
      'mom_reversal',
      'mom_short_long',
      'mom_1w',
    ],
  },
  {
    name: '波动因子',
    description: '收益率标准差与波动特征，衡量风险水平',
    factors: [
      'volatility_4w',
      'volatility_8w',
      'volatility_12w',
      'volatility_24w',
      'downside_vol_8w',
      'vol_ratio_4_12',
      'vol_change_4w',
      'realized_var_8w',
    ],
  },
  {
    name: '流动性因子',
    description: '成交量与成交额，反映交易活跃度',
    factors: [
      'avg_volume_4w',
      'avg_volume_8w',
      'avg_volume_12w',
      'volume_change_1w',
      'avg_amount_4w',
    ],
  },
  {
    name: '技术指标',
    description: 'RSI、KDJ、布林带、威廉指标等技术分析指标',
    factors: [
      'rsi_7',
      'rsi_14',
      'rsi_21',
      'kdj_k',
      'kdj_d',
      'kdj_j',
      'boll_width',
      'boll_pct',
      'willr_14',
      'cci_14',
    ],
  },
  {
    name: '均线因子',
    description: '均线比值、斜率与趋势强度，反映价格与均线关系',
    factors: [
      'ma_ratio_4w',
      'ma_ratio_8w',
      'ma_ratio_12w',
      'ma_ratio_24w',
      'ma_slope_4w',
      'ma_slope_12w',
      'ma_slope_24w',
      'ema_ratio_12',
    ],
  },
  {
    name: '风险因子',
    description: '最大回撤、偏度、峰度等风险特征指标',
    factors: [
      'max_retreat_4w',
      'max_retreat_8w',
      'max_retreat_12w',
      'skewness_8w',
      'kurtosis_12w',
      'ret_quantile_25_8w',
    ],
  },
  {
    name: '量价因子',
    description: '量价背离、上涨比例等量价关系指标',
    factors: [
      'obv_change_4w',
      'vol_price_corr_8w',
      'up_ratio_4w',
      'up_ratio_12w',
      'vol_up_down_ratio',
    ],
  },
]

/**
 * 图表说明文字组件
 *
 * @param text - 说明文字内容
 */
function ChartDescription({ text }: { text: string }) {
  return (
    <p
      style={{
        fontSize: '13px',
        color: '#86868B',
        lineHeight: 1.6,
        marginTop: '16px',
        paddingTop: '12px',
        borderTop: '1px solid #F5F5F7',
      }}
    >
      {text}
    </p>
  )
}

function FactorExplorer() {
  /** 当前选中的因子名称 */
  const [selectedFactor, setSelectedFactor] = useState<string>('mom_1m')
  /** 数据加载状态 */
  const [loading, setLoading] = useState(true)
  /** 从JSON加载的因子数据 */
  const [factorData, setFactorData] = useState<FactorData | null>(null)
  /** 数据加载错误信息 */
  const [error, setError] = useState<string | null>(null)

  /** 组件加载时从静态JSON文件获取因子数据 */
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        const response = await fetch('/data/factor_data.json')
        if (!response.ok) {
          throw new Error(`请求失败: ${response.status}`)
        }
        const data: FactorData = await response.json()
        setFactorData(data)
        // 如果默认因子不在数据中，取第一个因子
        if (data.factors.length > 0 && !data.factors.includes(selectedFactor)) {
          setSelectedFactor(data.factors[0])
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : '未知错误'
        console.error('加载因子数据失败:', message)
        setError(message)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  /** 因子与收益相关性列表，从加载的数据中获取 */
  const correlations = useMemo(() => {
    if (!factorData) return []
    return factorData.returnCorrelation.map((item) => ({
      factor_name: item.factor,
      correlation: item.correlation,
    }))
  }, [factorData])

  /** 因子IC值列表，从加载的数据中获取 */
  const icData = useMemo(() => {
    if (!factorData) return []
    return factorData.returnCorrelation.map((item) => ({
      factor_name: item.factor,
      ic_value: item.ic,
    }))
  }, [factorData])

  /** IC值柱状图数据 */
  const icBarData = useMemo(() => {
    return icData.map((ic) => ({
      name: ic.factor_name,
      value: ic.ic_value,
    }))
  }, [icData])

  /** 当前选中因子的散点数据，从scatterDataByFactor中读取，无数据时返回空数组 */
  const scatterData = useMemo(() => {
    if (!factorData) return []
    const factorScatter = factorData.scatterDataByFactor[selectedFactor]
    if (!factorScatter || factorScatter.length === 0) return []
    return factorScatter.map((point) => [point.x, point.y] as [number, number])
  }, [selectedFactor, factorData])

  /** 当前选中因子是否有散点数据 */
  const hasScatterData = scatterData.length > 0

  /** 散点图配置 */
  const scatterOption: EChartsOption = useMemo(() => {
    return {
      grid: {
        top: 40,
        right: 40,
        bottom: 60,
        left: 60,
      },
      tooltip: {
        trigger: 'item',
        formatter: (params: unknown) => {
          const p = params as { data: [number, number] }
          return `因子值: ${p.data[0].toFixed(3)}<br/>收益率: ${(p.data[1] * 100).toFixed(2)}%`
        },
      },
      xAxis: {
        name: `${selectedFactor} 值`,
        nameLocation: 'middle',
        nameGap: 30,
        type: 'value',
        splitLine: { lineStyle: { color: '#E8E8ED', type: 'dashed' } },
        axisLine: { lineStyle: { color: '#86868B' } },
        axisLabel: { color: '#86868B' },
      },
      yAxis: {
        name: '未来一周收益率',
        nameLocation: 'middle',
        nameGap: 40,
        type: 'value',
        axisLabel: {
          color: '#86868B',
          formatter: (value: number) => `${(value * 100).toFixed(0)}%`,
        },
        splitLine: { lineStyle: { color: '#E8E8ED', type: 'dashed' } },
        axisLine: { lineStyle: { color: '#86868B' } },
      },
      series: [
        {
          type: 'scatter',
          data: scatterData,
          symbolSize: 8,
          itemStyle: {
            color: '#0071E3',
            opacity: 0.6,
          },
          emphasis: {
            itemStyle: {
              color: '#0071E3',
              opacity: 1,
              borderColor: '#FFFFFF',
              borderWidth: 2,
            },
          },
        },
      ],
    }
  }, [scatterData, selectedFactor])

  /** 因子相关矩阵热力图配置，基于52x52相关系数矩阵生成 */
  const correlationHeatmapOption: EChartsOption = useMemo(() => {
    if (!factorData || !factorData.correlationMatrix.length) return {}
    const factors = factorData.factors
    const matrix = factorData.correlationMatrix
    const heatmapData: [number, number, number][] = []
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[i].length; j++) {
        heatmapData.push([j, i, matrix[i][j]])
      }
    }
    return {
      tooltip: {
        formatter: (params: unknown) => {
          const p = params as { data: [number, number, number] }
          const [x, y, val] = p.data
          return `${factors[x]} vs ${factors[y]}<br/>相关系数: ${val.toFixed(4)}`
        },
      },
      grid: { top: 10, right: 80, bottom: 80, left: 100 },
      xAxis: {
        type: 'category',
        data: factors,
        axisLabel: { rotate: 45, fontSize: 9, color: '#86868B' },
        splitArea: { show: true },
      },
      yAxis: {
        type: 'category',
        data: factors,
        axisLabel: { rotate: 45, fontSize: 9, color: '#86868B' },
        splitArea: { show: true },
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: true,
        orient: 'vertical',
        right: 10,
        top: 'center',
        inRange: { color: ['#FF3B30', '#FFFFFF', '#0071E3'] },
      },
      series: [
        {
          type: 'heatmap',
          data: heatmapData,
          label: { show: false },
          emphasis: {
            itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' },
          },
        },
      ],
    }
  }, [factorData])

  return (
    <div className="page-container">
      {/* 页面标题 */}
      <ScrollReveal index={0}>
        <div style={{ marginBottom: '32px' }}>
          <h1 className="page-title">因子探索</h1>
          <p style={{ fontSize: '15px', color: 'var(--color-text-subtle)' }}>
            深入分析各因子与未来收益的相关性
          </p>
        </div>
      </ScrollReveal>

      {/* 因子选择器 */}
      <ScrollReveal index={1}>
        <motion.div
          className="card"
          style={{ padding: '24px 32px', marginBottom: '24px' }}
          whileHover={{ y: -1 }}
          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '14px', color: 'var(--color-text-muted)' }}>选择因子：</span>
            {loading ? (
              <span style={{ color: 'var(--color-text-muted)' }}>加载中...</span>
            ) : error ? (
              <span style={{ color: '#FF3B30' }}>数据加载失败: {error}</span>
            ) : (
              <select
                value={selectedFactor}
                onChange={(e) => setSelectedFactor(e.target.value)}
                className="input"
                style={{ minWidth: '200px' }}
              >
                {factorData?.factors.map((f) => (
                  <option key={f} value={f}>
                    {f}{factorData.scatterDataByFactor[f] ? '' : ' (无散点数据)'}
                  </option>
                ))}
              </select>
            )}
          </div>
        </motion.div>
      </ScrollReveal>

      {/* 因子分布散点图 + IC值 */}
      <ScrollReveal index={2}>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: '24px',
            marginBottom: '24px',
          }}
        >
          <motion.div
            className="card"
            whileHover={{ y: -2 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
          >
            <div className="card-header">因子分布散点图</div>
            <div style={{ padding: '32px' }}>
              {loading ? (
                <div
                  style={{
                    height: '400px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#86868B',
                  }}
                >
                  加载中...
                </div>
              ) : !hasScatterData ? (
                <div
                  style={{
                    height: '400px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#86868B',
                    fontSize: '14px',
                  }}
                >
                  暂无散点数据
                </div>
              ) : (
                <>
                  <ReactECharts
                    option={scatterOption}
                    style={{ width: '100%', height: '400px' }}
                    opts={{ renderer: 'canvas' }}
                  />
                  <ChartDescription text="散点图展示因子值与未来一周收益率的关系。点的分布趋势反映因子与收益的相关性：右上倾斜表示正相关，右下倾斜表示负相关。" />
                </>
              )}
            </div>
          </motion.div>

          <motion.div
            className="card"
            whileHover={{ y: -2 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
          >
            <div className="card-header">因子IC值排名</div>
            <div style={{ padding: '32px' }}>
              {loading ? (
                <div
                  style={{
                    height: '400px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#86868B',
                  }}
                >
                  加载中...
                </div>
              ) : (
                <>
                  <BarChart data={icBarData} horizontal height="400px" showLabel />
                  <ChartDescription text="IC值（信息系数）衡量因子与收益的线性相关程度，绝对值越大说明因子预测能力越强。正值表示因子越大收益越高，负值表示因子越大收益越低。" />
                </>
              )}
            </div>
          </motion.div>
        </div>
      </ScrollReveal>

      {/* 因子与未来收益相关性明细 */}
      <ScrollReveal index={3}>
        <motion.div
          className="card"
          whileHover={{ y: -2 }}
          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
        >
          <div className="card-header">因子与未来收益相关性明细</div>
          <div style={{ padding: '32px' }}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #E8E8ED' }}>
                    <th
                      style={{
                        padding: '12px 16px',
                        textAlign: 'left',
                        fontSize: '12px',
                        fontWeight: 600,
                        color: '#86868B',
                      }}
                    >
                      因子名称
                    </th>
                    <th
                      style={{
                        padding: '12px 16px',
                        textAlign: 'right',
                        fontSize: '12px',
                        fontWeight: 600,
                        color: '#86868B',
                      }}
                    >
                      相关系数
                    </th>
                    <th
                      style={{
                        padding: '12px 16px',
                        textAlign: 'right',
                        fontSize: '12px',
                        fontWeight: 600,
                        color: '#86868B',
                      }}
                    >
                      IC值
                    </th>
                    <th
                      style={{
                        padding: '12px 16px',
                        textAlign: 'center',
                        fontSize: '12px',
                        fontWeight: 600,
                        color: '#86868B',
                      }}
                    >
                      信号强度
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {correlations.map((corr) => {
                    const ic = icData.find((i) => i.factor_name === corr.factor_name)
                    const signalStrength = Math.abs(ic?.ic_value || 0)
                    let strengthText = '弱'
                    let strengthColor = '#86868B'
                    if (signalStrength > 0.05) {
                      strengthText = '中'
                      strengthColor = '#0071E3'
                    }
                    if (signalStrength > 0.1) {
                      strengthText = '强'
                      strengthColor = '#34C759'
                    }
                    return (
                      <tr key={corr.factor_name} style={{ borderBottom: '1px solid #F5F5F7' }}>
                        <td style={{ padding: '12px 16px', fontSize: '14px', color: '#1D1D1F' }}>
                          {corr.factor_name}
                        </td>
                        <td
                          style={{
                            padding: '12px 16px',
                            textAlign: 'right',
                            fontSize: '14px',
                            color: corr.correlation >= 0 ? '#34C759' : '#FF3B30',
                          }}
                        >
                          {corr.correlation >= 0 ? '+' : ''}
                          {corr.correlation.toFixed(4)}
                        </td>
                        <td
                          style={{
                            padding: '12px 16px',
                            textAlign: 'right',
                            fontSize: '14px',
                            color: (ic?.ic_value || 0) >= 0 ? '#34C759' : '#FF3B30',
                          }}
                        >
                          {(ic?.ic_value || 0) >= 0 ? '+' : ''}
                          {(ic?.ic_value || 0).toFixed(4)}
                        </td>
                        <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                          <span
                            style={{
                              display: 'inline-block',
                              padding: '4px 12px',
                              borderRadius: '980px',
                              fontSize: '12px',
                              fontWeight: 500,
                              background: `${strengthColor}15`,
                              color: strengthColor,
                            }}
                          >
                            {strengthText}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      </ScrollReveal>

      {/* 因子相关矩阵热力图 */}
      <ScrollReveal index={4}>
        <motion.div
          className="card"
          style={{ marginBottom: '24px' }}
          whileHover={{ y: -2 }}
          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
        >
          <div className="card-header">因子相关矩阵热力图</div>
          <div style={{ padding: '32px' }}>
            {loading ? (
              <div
                style={{
                  height: '600px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#86868B',
                }}
              >
                加载中...
              </div>
            ) : !factorData ||
              !factorData.correlationMatrix ||
              factorData.correlationMatrix.length === 0 ? (
              <div
                style={{
                  height: '600px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#86868B',
                  fontSize: '14px',
                }}
              >
                暂无相关矩阵数据
              </div>
            ) : (
              <>
                <ReactECharts
                  option={correlationHeatmapOption}
                  style={{ width: '100%', height: '600px' }}
                  opts={{ renderer: 'canvas' }}
                />
                <ChartDescription text="热力图展示52个因子之间的相关系数矩阵。红色表示负相关，蓝色表示正相关，白色表示无相关。高度相关的因子组合可能存在共线性问题，在建模时应考虑剔除冗余因子。" />
              </>
            )}
          </div>
        </motion.div>
      </ScrollReveal>

      {/* 因子说明 */}
      <ScrollReveal index={5}>
        <motion.div
          className="card"
          whileHover={{ y: -2 }}
          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
        >
          <div className="card-header">因子说明</div>
          <div style={{ padding: '32px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '20px' }}>
              {FACTOR_CATEGORIES.map((category) => (
                <motion.div
                  key={category.name}
                  whileHover={{ y: -3, backgroundColor: 'rgba(0, 113, 227, 0.06)' }}
                  transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                  style={{
                    padding: '24px',
                    background: 'var(--color-bg-subtle)',
                    borderRadius: '16px',
                    cursor: 'default',
                  }}
                >
                  <h4
                    style={{
                      fontSize: '15px',
                      fontWeight: 600,
                      color: 'var(--color-text)',
                      marginBottom: '8px',
                    }}
                  >
                    {category.name}
                  </h4>
                  <p
                    style={{
                      fontSize: '13px',
                      color: 'var(--color-text-muted)',
                      lineHeight: 1.6,
                      marginBottom: '12px',
                    }}
                  >
                    {category.description}
                  </p>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                    {category.factors.map((factor) => (
                      <span
                        key={factor}
                        style={{
                          fontSize: '12px',
                          color: 'var(--color-primary)',
                          background: 'var(--color-primary-soft)',
                          padding: '4px 10px',
                          borderRadius: '6px',
                        }}
                      >
                        {factor}
                      </span>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </ScrollReveal>
    </div>
  )
}

export default FactorExplorer
