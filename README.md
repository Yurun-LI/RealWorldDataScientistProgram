# RealWorldDataScientistProgram
### Real-world Data Scientist Program (日本国立大学联合校企项目)
2020-09~2021-01
项目角色
Data Analyst
主要职责与业绩
目的: 对现有物流的配送路径的优化和分析,以及基于配送司机的路径学习 
数据来源: 日本西濃運輸株式会社(Seino Transportation Co., Ltd.) 大垣分店 2019.6-2020.7
的配送信息 
具体课题内容: 
①. 数据处理: 基于Pandas,Numpy等库的数据清洗和特征工程,实现了数据的初步加工 
②. 路径优化分析: 基于遗传算法计算了配送的最佳路径,并通过k-mean考察了配送人数从28削减到10对平均配送距离的影响,发现削减人数在1-9名内,平均配送路程增幅很小,但当削减人数超过10名后,平均路程会成阶梯式增加,表明配送人员的削减幅度为个位数. 
③. 由于装卸货耗时未知,基于XGboost构建了配送时间的预测模型,并结合遗传算法构建了在配送守时率的限制条件下的路径优化算法.其结果表明了在守时率保持的情况下当前平均配送路径可从43718m下降为41734m. 
④. 基于多年配送经验的老手司机的路径数据,构建了一种CNN的实时路径预测模型,与遗传算法得到的优化解相比,该模型在1-10处的单程配送中具有良好的预测性能.
