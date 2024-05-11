from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Создаем байесовскую сеть
bn = BayesianNetwork()

# Добавляем переменные
gender = 'Gender'
intelligence = 'Intelligence'
pass_exam = 'PassExam'
bn.add_nodes_from([gender, intelligence, pass_exam])


bn.add_edge(gender, intelligence)
bn.add_edge(gender, pass_exam)
bn.add_edge(intelligence, pass_exam)

cpd_gender = TabularCPD(variable=gender, variable_card=2, values=[[0.5], [0.5]])
cpd_intelligence = TabularCPD(variable=intelligence, variable_card=2, values=[[0.3, 0.5], [0.7, 0.5]],
                              evidence=[gender], evidence_card=[2])
cpd_pass_exam = TabularCPD(variable=pass_exam, variable_card=2,
                           values=[[0.8, 0.3, 0.9, 0.4], [0.2, 0.7, 0.1, 0.6]],
                           evidence=[gender, intelligence], evidence_card=[2, 2])

bn.add_cpds(cpd_gender, cpd_intelligence, cpd_pass_exam)


inference = VariableElimination(bn)
result = inference.query(variables=[pass_exam], evidence={gender: 0, intelligence: 1})
print(result)
