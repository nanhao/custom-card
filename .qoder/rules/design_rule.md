---
trigger: model_decision
description: 做技术设计时触发
---
- 必须输出设计文档
- 设计文档放在 d:\projects\custom-card/design/{intention_id} 目录下，目录不存在时，创建之
- 对类文件进行设计时，只设计接口。不要设计私有方法
- 设计文档应该结构合理，内容太多时拆分成多个子文档
- 设计文档必须包含关键流程交互图。交互图用 plantuml 画，单独文件，不要嵌在文档中。文档中引用交互图
