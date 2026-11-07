"""Produce Chinese reports from completed, immutable experimental measurements only."""
from pathlib import Path
import statistics
from common import STUDY,now,read,rows,sha,write


def number(value,digits=3):
    return '未定义' if value is None else f'{value:.{digits}f}'


def percent(value):
    return '未定义' if value is None else f'{100*value:.2f}%'


def interval(result,percentage=True):
    scale=100 if percentage else 1
    unit=' 个百分点' if percentage else ''
    point=result['difference'];bounds=result['interval']
    if point is None:
        return '未定义；请查缺失分母'
    text=f'{point*scale:+.2f}{unit}'
    if bounds is not None:
        text+=f' [{bounds[0]*scale:+.2f}, {bounds[1]*scale:+.2f}]'
    else:
        text+=' [区间未定义]'
    return text


def direction(result):
    bounds=result['interval']
    if result['difference'] is None or bounds is None:
        return '当前缺失情况不足以给出这个完整对照的估计。'
    if bounds[0]>0:
        return '在当前数据与机器评分口径下，区间支持前者更高。'
    if bounds[1]<0:
        return '在当前数据与机器评分口径下，区间支持前者更低。'
    return '区间包含零，当前结果不能确定差异方向，也不能据此证明两者没有差异。'


def write_u_group_report(states,summaries,reports):
    names={'presentation':'可见呈现','age_band':'外观年龄','skin_tone':'可见肤色'}
    lines=['# 理解端的群体表现与图像依赖','',
        '直接问答只纳入已有可靠参考标签的图片。正确率是两名评审共同确认模型答案表达了该参考类别的图片数，除以该类别全部计划问答图片数；评分失败不算确认正确。训练组报告三个种子的平均值，括号 n 是每个种子的参考图片数，不把三次使用同一图片当成三个独立人物。',
        '', '群体差距为同一来源、同一属性各已知类别准确率的最大值减最小值。表中给出各训练种子差距的平均值及最小—最大值；这不是置信区间。只有一个已知类别时差距未定义。差距减小可能伴随整体能力下降，必须同时看准确率。不同群体的场景内容与可见性并未完全匹配，因此这里是描述性公平性证据。',
        '', '| 训练组 | 来源 | 属性 | 各类别准确率（每种子图片数） | 整体准确率 | 群体差距均值（百分点；种子范围） |',
        '|---|---|---|---|---:|---:|']
    sources=sorted({g['source'] for summary in summaries.values() for g in summary['u']['groups'] if g['task']=='qa'})
    for arm in ('base','T0','T1','T2','T3'):
        selected=[summaries[s['id']] for s in states if s['arm']==arm]
        for source in sources:
            for field,label in names.items():
                groups=[g for summary in selected for g in summary['u']['groups']
                    if g['task']=='qa' and g['source']==source and g['field']==field]
                if not groups:
                    continue
                classes=groups[0]['appearance_subgroups'][field]['classes'];cells=[]
                for category in classes:
                    records=[g['appearance_subgroups'][field]['classes'][category] for g in groups]
                    values=[r['coverage']['value'] for r in records if r['coverage']['value'] is not None]
                    counts={r['cases'] for r in records}
                    if len(counts)!=1:
                        raise ValueError('Group reference denominator changed across training seeds')
                    cells.append(category+': '+percent(statistics.mean(values) if values else None)+f'（n={counts.pop()}）')
                gaps=[g['appearance_subgroups'][field]['descriptive_max_min_gap'] for g in groups
                    if g['appearance_subgroups'][field]['descriptive_max_min_gap'] is not None]
                scores=[g['common_image_mean_coverage']['value'] for g in groups]
                gap=number(100*statistics.mean(gaps),2)+f'（{number(100*min(gaps),2)}—{number(100*max(gaps),2)}）' if gaps else '未定义'
                lines.append(f'| {arm} | {source} | {label} | '+'; '.join(cells)+f' | {percent(statistics.mean(scores))} | {gap} |')
    lines += ['', '## 图像依赖对照','',
        '每对图片使用相同问题但具有不同的已知参考类别。成功要求两张图的答案都被共同确认正确；固定答案不能使一对样本都正确。该对照改变整张图，不是只改变单一人口属性的因果反事实。表中分开列出全部配对与内容来源相同的配对；三种子的平均正确率使用每种子相同的配对分母。',
        '', '| 训练组 | 属性 | 两张图均正确 / 全部配对 | 两张图均正确 / 同内容配对 |',
        '|---|---|---:|---:|']
    for arm in ('base','T0','T1','T2','T3'):
        selected=[summaries[s['id']] for s in states if s['arm']==arm]
        for field,label in names.items():
            cells=[]
            for scope in ('all','same_content'):
                records=[s['u']['image_dependence_summary'][field][scope] for s in selected]
                values=[r['value'] for r in records if r['value'] is not None]
                counts={r['denominator'] for r in records}
                if len(counts)!=1:
                    raise ValueError('Image-pair denominator changed')
                cells.append(percent(statistics.mean(values) if values else None)+f'（配对数={counts.pop()}）')
            lines.append(f'| {arm} | {label} | '+' | '.join(cells)+' |')
    lines += ['', '相同问题的原生无图回答及其实际断言/不确定性保存在每状态 `summary.json` 的 `u.no_image_controls`。合理拒答不是公平性失败；不能把无图猜不中标签解释为理解能力下降。',
        '', '所有直接问答结果均以当前机器核验事实为参照，人工人口外观金标准尚未建立。标签未知的图片数和字段覆盖见冻结的数据审计；这些缺失样本没有计入已知群体正确率。']
    (reports/'u_group_results.md').write_text('\n'.join(lines)+'\n')


def write_g_quality_report(states,summaries,reports):
    lines=['# 生成端的内容符合与质量','',
        '各率均以全部计划请求为分母：显式请求每状态 288 条，中性请求每状态 192 条。内容/人物指标采用两名评审共同确认，不确定、分歧和评分失败不算成功。人物比例仅为输出有效性代理，不能代替完整画质评价。',
        '', '| 状态 | 显式：可辨认单人 | 显式：场景符合 | 显式：动作符合 | 显式：物体符合 | 中性：可辨认单人 | 中性：场景符合 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for state in states:
        outcomes=summaries[state['id']]['g']['common'];explicit=outcomes['explicit'];neutral=outcomes['neutral']
        values=[explicit['valid_person']['value'],*(explicit['content'][k]['value'] for k in ('setting','action','object')),
            neutral['valid_person']['value'],neutral['content']['setting']['value']]
        lines.append('| '+state['id']+' | '+' | '.join(percent(v) for v in values)+' |')
    lines += ['', '## 明显结构错误的独立机器判断','',
        '分母为每状态全部 480 张图。每格依次列“确认有明显结构错误的张数 / 480；不确定或未评分张数”。未知不视为无错误。此项没有独立人工画质参考，也不评价吸引力或以人口类别评价质量。',
        '', '| 状态 | Qwen | Ministral |','|---|---:|---:|']
    for state in states:
        cells=[]
        for judge in ('qwen','ministral'):
            groups=[summaries[state['id']]['g'][judge][kind] for kind in ('explicit','neutral')]
            errors=sum(g['major_anatomy_error'].get('YES',0) for g in groups)
            unknown=sum(g['major_anatomy_error'].get('UNCERTAIN',0)+g['major_anatomy_error'].get('unscored',0) for g in groups)
            denominator=sum(g['planned'] for g in groups)
            cells.append(f'{errors}/{denominator}；未知 {unknown}')
        lines.append('| '+state['id']+' | '+' | '.join(cells)+' |')
    lines += ['', '图片类型、构图、场景细节可见性、两评审分歧及各职业每个类别的原始计数均保存在各状态 `aggregate/summary.json`。中性类别比例变化必须结合这里的有效人物、场景和结构错误观察，不能以人数分布更均衡替代任务能力保持。']
    (reports/'g_quality_results.md').write_text('\n'.join(lines)+'\n')


def main():
    config=STUDY/'configs/formal_v1';root=STUDY/'results/formal_v1';reports=STUDY/'reports'
    states=rows(config/'states.jsonl');freeze=read(config/'freeze.json')
    if read(root/'final_integrity.json')['status']!='passed':
        raise RuntimeError('Final output integrity must pass before writing result reports')
    if freeze.get('grading_location')=='remote_gpu' and read(
            Path(freeze['remote_grading_queue_status']))['status']!='completed':
        raise RuntimeError('All remote formal grades must finish before writing result reports')
    for gpu in (0,1):
        if read(STUDY/f'results/formal_worker_v1_gpu{gpu}.json')['status']!='completed':
            raise RuntimeError('Every formal workload must complete before results reports are written')
    summaries={s['id']:read(Path(s['aggregate'])/'summary.json') for s in states}
    comparisons={name:read(root/'comparisons'/(name+'.json')) for name in
        ('T1-T0','T2-T1','T3-T2','T0-base','T1-base','T2-base','T3-base')}
    audit=read(config/'data_audit.json');quality=read(freeze['base_quality_review'])
    write_u_group_report(states,summaries,reports)
    write_g_quality_report(states,summaries,reports)
    main_lines=['# BAGEL 细节理解监督实验：实际结果','',
        f'本报告生成于 {now()}。所有数值来自本轮完成的真实模型运行；未用模拟样例代替研究结果。',
        '', '模型为 BAGEL-7B-MoT。基座与 T0–T3 四组比较，各训练组包含 7101、7102、7103 三个独立种子，固定最后一轮检查点。仅训练图片→文字损失，更新同一组 U 专家 LoRA 参数。',
        f'实际训练图片 {audit["train"]["images"]} 张、开发图片 {audit["dev"]["images"]} 张、测试图片 {audit["test"]["images"]} 张。训练图片及权重在四组间相同。',
        '', 'T0 只给简短的可见活动/主体描述；T1 给普通视觉细节；T2 在保持共同动作、物体和背景事实时，以可靠人口外观事实替换部分附加细节；T3 将部分 T2 描述拆成互不读取答案的独立问答。BPO 以肖像为主，来源职业不能充当真实职业标签，因此本轮不是原论文职业描述训练的严格复现。',
        '', '## 理解端', '',
        '覆盖率是回答正确表达的已核验参考事实占比，先在每张图片内计算，再在同来源图片间平均。下面采用两名评审共同确认；评审失败不会被当作成功。断言准确率由两名评审分别给出，是回答中有图像证据支持的断言数除以全部断言数；空回答没有定义该准确率。未知人口标签没有被补全。',
        '', '| 状态 | 来源 | 普通事实覆盖率 | 人口事实覆盖率（有参考事实的图片数） | 断言准确率 Qwen / Ministral |',
        '|---|---|---:|---:|---:|']
    for state in states:
        for group in summaries[state['id']]['u']['groups']:
            if group['task']!='detailed':
                continue
            ordinary=group['common_groups']['ordinary'];population=group['common_groups']['demographic']
            precisions=[group['judges'][j]['image_mean_claim_precision'] for j in ('qwen','ministral')]
            main_lines.append(f'| {state["id"]} | {group["source"]} | {percent(ordinary["value"])} | {percent(population["value"])}（{population["defined"]}/{group["planned_cases"]}） | '+
                ' / '.join(percent(p['value'])+f'（n={p["defined"]}）' for p in precisions)+' |')
    main_lines += ['', '各属性直接问答、分组准确率、群体差距及换图对照的可读表格见 [理解端群体结果](u_group_results.md)。原始无图回答、截断及空回答统计保存在各状态的 `aggregate/summary.json`。来源职业留出不等于人物身份留出；各群体内容和可见性不同，描述性差距不能单独解释为人口属性的因果作用。',
        '', '## 原生生成端', '',
        '每个状态有 288 条显式属性/动作/物体/场景请求，以及 192 条中立职业请求。所有状态使用相同原始提示和实际初始噪声；官方 1024×1024、50 个时间点及引导设置保持不变。没有输入理解端答案、改写计划或筛选后替换图片。',
        '', '严格联合成功要求单一可辨认的近似真人照片、全部三个人口外观类别及全部动作/物体/场景要求，均被两名评审确认。分母为全部 288 条请求，不确定、分歧和评分失败不算确认成功。人口类别是可见呈现、外观年龄及实际光照下肤色的机器代理，不是人物身份或真实年龄。',
        '', '| 状态 | 严格联合成功 / 288 | 呈现符合率 | 外观年龄符合率 | 肤色符合率 |',
        '|---|---:|---:|---:|---:|']
    for state in states:
        outcome=summaries[state['id']]['g']['common']['explicit'];success=outcome['joint_success']
        main_lines.append(f'| {state["id"]} | {success["numerator"]}/{success["denominator"]}（{percent(success["value"])}） | '+
            ' | '.join(percent(outcome['attribute_success'][f]['value']) for f in ('presentation','age_band','skin_tone'))+' |')
    main_lines += ['', '中性请求的群体差异 RD，是每个职业内已知类别比例两两绝对差的平均，再对八个职业平均。越小仅表示已判定类别的比例越均衡。二类和三类的量纲不同，不能直接横向比较。括号为已判定人数/192；任一职业没有已知标签时，完整职业均值未定义。',
        '', '| 状态 | 呈现 RD（已知数） | 外观年龄 RD（已知数） | 肤色 RD（已知数） |',
        '|---|---:|---:|---:|']
    for state in states:
        outcome=summaries[state['id']]['g']['common']['neutral'];cells=[]
        for field in ('presentation','age_band','skin_tone'):
            known=sum(c[field]['known'] for c in outcome['by_occupation'].values())
            cells.append(number(outcome['mean_rd'][field]['full_occupation_mean'])+f'（{known}/{outcome["planned"]}）')
        main_lines.append('| '+state['id']+' | '+' | '.join(cells)+' |')
    main_lines += ['', '人数比例均衡不能替代画质或指令遵从。可读的内容与质量对照见 [生成端内容与质量](g_quality_results.md)。无人/多人、图片类型、构图、已知/未知比例及两名评审的分别结果，完整保存在原始汇总中。',
        '', '## 主要配对对照', '',
        '下表是严格联合生成成功率的差值，方括号为 95% 配对重采样区间；同时重采样训练种子、场景和实际噪声，保留同一噪声下全部 18 个属性组合。仅有三个训练种子和八个场景，不能将区间解释为覆盖所有训练不确定性。T2−T1 是主要比较，其余为次要/探索性比较。',
        '', '| 比较（前者减后者） | 差值及 95% 区间 |', '|---|---:|']
    for name,comparison in comparisons.items():
        main_lines.append(f'| {name} | {interval(comparison["g_explicit"]["common"]["joint_success"])} |')
    main_lines += ['', f'原始逐图片记录与评分：`{root}`。完整配对区间：`{root / "comparisons"}`。每状态的 `aggregate/summary.json` 包含所有种子与缺失分母；没有选择最佳种子。']
    (reports/'main_results.md').write_text('\n'.join(main_lines)+'\n')
    findings=['# 实验分析','', '## 1. 增加可靠人口外观事实后，理解端是否学到更多？','',
        '问题与对照：T2 对 T1 保留共同内容事实，替换部分附加普通细节，图片和训练配置相同。下面是两名评审共同确认的人口事实覆盖率变化；它检验已知参考事实的表达，并不等于所有群体差距都减小。','']
    for source,metrics in comparisons['T2-T1']['u'].items():
        result=metrics['detailed:coverage:demographic']
        findings.append(f'- {source}：{interval(result)}。{direction(result)}')
    findings += ['', 'U 结果还必须结合普通细节覆盖率、断言准确率、直接问答和换图对照判断。更长的回答或更熟悉的表达形式，本身不能证明视觉依赖或公平性改善。',
        '', '## 2. 同一干预是否改变原生生成？','']
    primary=comparisons['T2-T1']['g_explicit']['common']['joint_success']
    versus_base=comparisons['T2-base']['g_explicit']['common']['joint_success']
    findings += [f'实际结果：T2−T1 的严格联合生成成功率差为 {interval(primary)}。{direction(primary)}',
        f'与未训练基座相比，T2 的差值为 {interval(versus_base)}。{direction(versus_base)}',
        '判断：相对一个退化的训练对照更好，可能只是能力保持；只有同时与原始基座比较，才能区分这种情况与真实增益。即使 U 和 G 都改善，本轮也未通过中介或内部路径干预证明两端使用了同一段偏差计算。',
        '', '## 3. 问答形式是否提供额外收益？','',
        f'实际结果：T3−T2 的严格联合生成成功率差为 {interval(comparisons["T3-T2"]["g_explicit"]["common"]["joint_success"])}。'+direction(comparisons['T3-T2']['g_explicit']['common']['joint_success']),
        '判断：这是独立问答形式的对照。各组图片曝光相同，但问答上下文数量和输入 token 不同；目标 token 的实际差异必须同时报告，不能声称算力完全相同。',
        '', '## 4. 当前证据的边界','',
        f'- 基座质量检查使用全部 10 张固定样图，两名机器评审共同确认可辨认人物 {quality["confirmed_people"]}/10、请求场景 {quality["confirmed_settings"]}/10。该步骤为自动检查，没有冒充人工视觉金标准。',
        f'- 基座技术样图完整可解析的双评审为 {quality["jointly_scored"]}/10。一条图片类型评分格式失败保留为未知；正式前将质量放行规则从所有评分必须可解析，修订为全部尝试齐全且至少 90% 可解析，并要求十张原图均有实际代理查看记录。旧失败保留，评分解析器、正式指标和分母未改，见 `configs/base_quality_review_revision_v2.json`。',
        '- 另对这 10 张原图做了逐图代理目视检查：三张有额外人物，木工样图出现明显不合场景的鱼形物体，部分工具和动作不清楚。样图均保留，未按质量换图；该有限检查不能估计全部输出的质量比例。逐图记录见 `results/base_quality_v1/codex_visual_review_v1.json`。',
        '- 技术检查曾发现两名评审共同把雕像当成人物并赋予肤色。原错误保留，之后加入单独图片类型检查；这仍不能保证所有人口类别都准确。',
        '- BPO 来源以肖像为主，场景细节来自额外的公开 COCO 图片；两种来源分开报告。图片来自公开数据并不保证未见于基础模型预训练。',
        '- 所有未知标签、失败评分和群体样本量保留。已知类别比例更均衡，不代表未知样本同样均衡，更不是现实职业人口统计。',
        '- 本轮没有定位具体层或证明偏差只发生在 G 端。它回答监督内容/形式的受控训练差异与原生 G 表现之间的关系。']
    (reports/'findings_and_analysis.md').write_text('\n'.join(findings)+'\n')
    ablations=['# 对照与机制核验','',
        '实际完成的受控训练对照是 T1−T0（普通细节）、T2−T1（可靠人口外观事实）和 T3−T2（独立问答）。每个训练组都有三个种子，没有按结果择优。','',
        '| 对照 | 严格联合 G 成功率变化及 95% 区间 |','|---|---:|']
    for name in ('T1-T0','T2-T1','T3-T2'):
        ablations.append(f'| {name} | {interval(comparisons[name]["g_explicit"]["common"]["joint_success"])} |')
    ablations += ['', '实际技术核验包括：U 专家 LoRA 的非零有限梯度、冻结参数未变化、问答分支隔离、检查点重载、同一实际噪声生成复现。U 参数更新改变了 G 文本条件缓存，证明路径可以被影响；这不是人口偏差已经被纠正的证据。',
        '', '本轮未运行新的层撤销、纠错训练、公平损失或生成筛选。这些候选机制不能写成已完成的消融或机制结论。']
    (reports/'ablation_results.md').write_text('\n'.join(ablations)+'\n')
    (reports/'next_steps.md').write_text('''# 后续判断

1. 优先对当前已生成样本做独立人工事实/人口外观复核，按已知、未知与评审分歧分层取样，核对机器指标是否存在群体差异。不能直接用两名机器一致替代人工参考。
2. 根据本轮 U 覆盖、准确率及换图对照判断源端是否真正学到图像事实。若源端缺乏清晰收益，先核查事实可见性、训练分布和学习是否充分，不直接叠加新模块；若源端有清晰收益而 G 证据仍弱，再做同噪声的内部条件表示干预。
3. 只有需要更精确地区分本轮区间内的可能差异时，才另行预先冻结更多场景、噪声与训练种子的重复实验。不能将区间包含零直接解释为完全没有迁移。

以上是基于本轮证据的后续候选，不冒充已完成实验，也未自动改变本轮冻结合同。
''')
    protocol=(config/'protocol_at_freeze.md').read_text()
    protocol += '\n\n## 实际完成记录\n\n'+f'冻结文件：`{config / "freeze.json"}`。训练/开发/测试图片数为 {audit["train"]["images"]}/{audit["dev"]["images"]}/{audit["test"]["images"]}。12 次正式训练及原始基座与 12 个训练状态的 U/G 评估均完成；每状态 480 张正式 G 图，共 6,240 张。\n\n'
    protocol += '实际目标 token 相对差异：'+', '.join(f'{k}：{v:+.2%}' for k,v in audit['train']['target_budget_relative_differences'].items())+'。T3 输入问题和上下文成本单列于 data_audit.json，不能称计算量完全相同。\n'
    (reports/'experiment_protocol.md').write_text(protocol)
    implementation=reports/'implementation_verification.md'
    implementation.write_text(implementation.read_text()+f'\n\n## 正式完成状态（{now()}）\n\n'+
        '前述准备阶段的待运行项目现已通过实际运行核验：来源事实、图片类型、修订后的固定评分对照、小实验五状态评分、两条正式原生 GPU 队列及全部正式外部评分均完成。旧失败队列和错误记录保留，不将它们改写为成功。真实完成状态与冻结输入见 `configs/formal_v1/freeze.json`；原始基座自动质量检查见 `results/base_quality_v1/quality_review.json`，逐图代理查看另见 `results/base_quality_v1/codex_visual_review_v1.json`，两者都不是人工人口属性金标准。\n')
    (STUDY/'README.md').write_text(f'''# BAGEL 细节理解监督与原生生成迁移

本轮正式核心实验已经实际完成：12 次训练，基座及 12 个训练状态的 U 评估和合计 6,240 张原生 G 图，两个独立机器评审及固定配对区间。所有种子、失败与未知分母保留。

- [实际结果](reports/main_results.md)
- [分析与局限](reports/findings_and_analysis.md)
- [实验协议](reports/experiment_protocol.md)
- [实现核验](reports/implementation_verification.md)
- [对照与机制核验](reports/ablation_results.md)
- [后续判断](reports/next_steps.md)
- [运行与接手手册](RUNNING.md)

原生训练/推理由原服务器两张卡执行；正式外部评分的实际位置及资格记录见冻结合同。运行状态可用 `./study.py status --remote` 核验。

完成报告生成时间：{now()}。报告由实际记录自动汇总；独立人工人口属性复核尚未完成。报告不能被理解为已经定位了具体偏差层或证明两端使用同一计算。
''')
    write(root/'report_manifest.json',dict(completed_utc=now(),source_sha256=sha(__file__),
        reports={str(path):sha(path) for path in reports.glob('*.md')},freeze_sha256=sha(config/'freeze.json'),
        status='actual_results_written',manual_result_review='pending_agent_review'))


if __name__=='__main__':
    main()
