import pandas as pd

csv_cursos = pd.read_csv('data/transformed_data/cursos.csv', sep=';', encoding='latin1', low_memory=False)
csv_ies = pd.read_csv('data/transformed_data/ies.csv', sep=';', encoding='latin1', low_memory=False)
csv_enem = pd.read_csv('data/transformed_data/enem.csv', sep=';', encoding='latin1', low_memory=False)

print('as colunas do csv de cursos sao:')
for col in csv_cursos.columns:
    print(col)
print('\nas colunas do csv de ies sao:')
for col in csv_ies.columns:
    print(col)
print('\nas colunas do csv do enem sao:')
for col in csv_enem.columns:
    print(col)