from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
# Создаем байесовскую сеть
bn = BayesianNetwork()

# Добавляем переменные
parents_income = "Parents Income"
cpd_parents_income = TabularCPD(variable=parents_income, variable_card=3, values=[[0.5], [0.35], [0.15]])


income = "Inc"
cpd_income = TabularCPD(variable=income, variable_card=3, values=[[0.2, 0.2, 0.03], [0.2, 0.6, 0.07], [0.6, 0.2, 0.9]],
                              evidence=[parents_income], evidence_card=[3])



assets = "Ass"
cpd_assets = TabularCPD(variable=assets, variable_card=3, values=[[0.5, 0.2, 0.2], [0.3, 0.5, 0.3], [0.2, 0.3, 0.5]],
                              evidence=[income], evidence_card=[3])

university = "Un"
cpd_university = TabularCPD(variable=university, variable_card=2, values=[[0.8], [0.2]])

future_income = "Future I"
cpd_future = TabularCPD(variable=future_income, variable_card=2, values=[[0.8, 0.5, 0.7, 0.4, 0.6, 0.3, 0.7, 0.4, 0.6, 0.3, 0.5, 0.2, 0.6, 0.3, 0.5, 0.2, 0.4, 0.1],
                                                                         [0.2, 0.5, 0.3, 0.6, 0.4, 0.7, 0.3, 0.6, 0.4, 0.7, 0.5, 0.8, 0.4, 0.7, 0.5, 0.8, 0.6, 0.9]],
                              evidence=[assets, income, university], evidence_card=[3, 3, 2])

gender = "Gender"
cpd_gender = TabularCPD(variable=gender, variable_card=2, values=[[0.5], [0.5]])

ratio = "DtI"
cpd_ratio = TabularCPD(variable=ratio, variable_card=2, values=[[0.7, 0], [0.3, 1]], evidence=[gender], evidence_card=[2])


age = "Age"
cpd_age = TabularCPD(variable=age, variable_card=3, values=[[0.3], [0.3], [0.4]])

payment = "Payment H"
cpd_payment = TabularCPD(variable=payment, variable_card=3, values=[[0.6, 0.7, 0.2, 0.3, 0.1, 0.1],
                                                                    [0.3, 0.2, 0.3, 0.3, 0.3, 0.4],
                                                                    [0.1, 0.1, 0.5, 0.4, 0.6, 0.5]],
                                                            evidence=[age, ratio], evidence_card=[3, 2])

reliable = "Reliability"
cpd_reliable = TabularCPD(variable=reliable, variable_card=2, values=[[0.98, 0.92, 0.9, 0.95, 0.9, 0.85, 0.9, 0.85, 0.75],
                                                                      [0.02, 0.08, 0.1, 0.05, 0.1, 0.15, 0.1, 0.15, 0.25]],
                                                            evidence=[payment, age], evidence_card=[3, 3])

credit = "Credit"
cpd_credit = TabularCPD(variable=credit, variable_card=2, values=[[0.8, 0.9, 0.6, 0.7, 0.3, 0.4, 0.1, 0.2],
                                                                  [0.2, 0.1, 0.4, 0.3, 0.7, 0.6, 0.9, 0.8]],
                                                            evidence=[reliable, future_income, ratio], evidence_card=[2, 2, 2])

bn.add_nodes_from([parents_income, income, assets, university, future_income, gender, ratio, age, payment, reliable, credit])

bn.add_edge(parents_income, income)

bn.add_edge(income, assets)

bn.add_edge(income, future_income)
bn.add_edge(assets, future_income)
bn.add_edge(university, future_income)

bn.add_edge(gender, ratio)

bn.add_edge(ratio, payment)
bn.add_edge(age, payment)

bn.add_edge(payment, reliable)
bn.add_edge(age, reliable)

bn.add_edge(reliable, credit)
bn.add_edge(future_income, credit)
bn.add_edge(ratio, credit)

bn.add_cpds(cpd_parents_income, cpd_income, cpd_assets, cpd_university, cpd_future, cpd_gender, cpd_ratio, cpd_age, cpd_payment, cpd_reliable, cpd_credit)

inference = VariableElimination(bn)

# result = inference.query(variables=[credit], evidence={age: 0, gender: 0, parents_income: 0, university: 0})


result = inference.query(variables=[university], evidence={credit: 1, parents_income: 2, age: 0})
result2 = inference.query(variables=[reliable], evidence={credit: 1, age: 0})
resultCredit = inference.query(variables=[credit], evidence={parents_income: 1, university: 0})

size = 10000
sampler = BayesianModelSampling(bn)
evidence1 = [State('Credit', 1), State('Parents Income', 2), State('Age', 0)]
samples1 = sampler.rejection_sample(evidence=evidence1, size=size,)



print('The probability of Mepi students: ', (float(samples1['Un'].value_counts().get(0)) / size) * 100, "%")
# print("SAMPLES2: \n\n\n")
# print(samples2)
# result2 = inference.query(variables=[credit], evidence={university: 1, parents_income: 1, age: 1})
# result3 = inference.query(variables=[credit], evidence={parents_income: 0, age: 1, gender: 0})
# result4 = inference.query(variables=[credit], evidence={university: 0, age:1})
# result4 = inference.query(variables=[credit], evidence={university: 0, age:})
# result4 = inference.query(variables=[credit], evidence={university: 0, age:})

# print(result)
# print(result2)
# print(result3)
# print(result4)