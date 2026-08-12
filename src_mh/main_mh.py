from src_mh.dados.carregar_dados import CarregarDados

if __name__ == "__main__":
    estrategia_modelo = CarregarDados(
        colunas=['Metragem', 'Quartos', 'Banheiros', 'Vagas', 'Bairro',
                 'Valor_da_Venda'])
    x_train_transformado, x_test_transformado, y_train, y_test, = estrategia_modelo.separar_treino_teste()
    print(x_test_transformado)
    print(y_test)