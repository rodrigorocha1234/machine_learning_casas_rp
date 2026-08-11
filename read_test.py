from src.service_site.extracao_lago_imobiliaria import ExtracaoLagoImobiliaria

eli = ExtracaoLagoImobiliaria(url='https://www.lagoimobiliaria.com.br/comprar/Ribeirao-Preto/Apartamento/Padrao/Campos-Eliseos/171029')

eli.abrir_site()
eli.obter_metragem()