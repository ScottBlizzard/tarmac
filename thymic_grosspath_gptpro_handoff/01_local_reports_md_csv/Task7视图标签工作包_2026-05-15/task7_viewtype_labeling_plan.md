# Task7视图类型标注模板说明

- 当前主图总数：285
- 优先标注清单总数：98
- P1 原始多图病例：17
- P2 Task7高危漏诊病例：41
- P3 高危正确对照：20
- P4 低危正确对照：20

建议视图标签字段：
- view_type_round1: cut_surface / outer_surface / mixed / unclear
- view_type_confidence: high / medium / low
- cut_surface_degree: none / weak / moderate / strong
- outer_surface_degree: none / weak / moderate / strong
- mixed_context: yes / no
- tumor_visible_degree: low / medium / high
- fat_context_degree: low / medium / high
- scale_visible: yes / no
- is_preferred_main_view: yes / no
- alternate_view_needed: yes / no

优先顺序：先标多图病例，再标Task7高危漏诊病例，最后补正确对照组。