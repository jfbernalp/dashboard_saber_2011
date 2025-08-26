import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import textwrap
import statsmodels.api as sm

# --- Configuración de la Página ---
st.set_page_config(page_title="Análisis Pruebas ICFES 2011", layout="wide")

# --- Paleta de Colores ---
# Extraída de la imagen para usar en los gráficos.
PALETA_TIERRA = ['#73AB84', '#A9D9B8', '#406A52', '#E3D1A7', '#D9A47F']
PALETA_COLORES = ["#5B97B1", "#F7B828", "#E8843A", "#1F3A4E", "#A5C3D9"]

# --- Título Principal ---
st.title('Análisis de Resultados de las Pruebas ICFES del Año 2011')

# --- Carga de Datos ---
DATA_URL = 'datos_dashboard.csv'

@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url)
        # Limpieza básica de datos
        df['ESTU_GENERACION_E_NORMALIZED'] = df['ESTU_GENERACION-E'].replace({
            'GENERACION E - EXCELENCIA NACIONAL': 'EXCELENCIA',
            'GENERACION E - EXCELENCIA DEPARTAMENTAL': 'EXCELENCIA',
            'GENERACION E - EQUIPO': 'EQUIPO'
        })
        mapeo_ingles = {
            'A-': 1,
            'A1': 2,
            'A2': 3,
            'B1': 4,
            'B+': 5
        }

        # Se crea la nueva columna numérica 'INGLES_NORMALIZADO'.
        df['INGLES_NORMALIZADO'] = df['DESEMP_INGLES'].map(mapeo_ingles)

        df['FAMI_ESTRATOVIVIENDA'] = df['FAMI_ESTRATOVIVIENDA'].str.replace('Estrato ', '')
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{url}'. Asegúrate de que está en la misma carpeta.")
        return None

icfes_df = load_data(DATA_URL)

if icfes_df is None:
    st.stop()

# --- Barra Lateral de Navegación ---
st.sidebar.title('Menú de Navegación')
opcion_seleccionada = st.sidebar.radio(
    'Selecciona una sección para visualizar:',
    ('Análisis Descriptivo', 'Análisis de Resultados de Inglés', 'Análisis Socioeconómico', 'Análisis de resultados por género', 'Análisis de Becados', 'Conclusiones')
)
st.sidebar.info("App creada a partir de un análisis exploratorio.")
# Normalizacion de columnas categoricas de interés
col_carne = 'FAMI_COMECARNEPESCADOHUEVO'
col_leche = 'FAMI_COMELECHEDERIVADOS'
col_puntaje = 'PUNT_GLOBAL'

# --- Contenido Principal (basado en la selección de la barra lateral) ---

# --- Sección 0: Análisis Descriptivo ---
if opcion_seleccionada == 'Análisis Descriptivo':
    st.header('Análisis inicial de los datos')
# analisis descriptivo de las variables numericas
    df_numerico = icfes_df.select_dtypes(include=np.number)# seleccionde las columnas numericas
    df_filtrado = df_numerico.loc[:, ~df_numerico.columns.str.contains('COD')]
    estadisticas = df_filtrado.describe()

    st.dataframe(estadisticas)

    st.subheader('Datos sobre la alimentación:')
# Analisis descriptivo de columnas categoricas
    conteo_carne = icfes_df[col_carne].value_counts()
    conteo_leche = icfes_df[col_leche].value_counts()
    df_conteos = pd.DataFrame({
        'Conteo_Carne_Pescado_Huevo': conteo_carne,
        'Conteo_Leche_Derivados': conteo_leche
    })

    df_conteos = df_conteos.fillna(0).astype(int)
    df_conteos.rename(index={'-': 'Sin Respuesta'}, inplace=True)
    df_conteos.loc['Total'] = df_conteos.sum()

    st.dataframe(df_conteos)

    # --- Preparar los Datos para Graficar ---
    # Excluimos las filas que no son categorías de consumo para el gráfico.
    # Usamos .drop() para eliminar las filas por su nombre de índice.
    df_para_grafico = df_conteos.drop(['Sin Respuesta', 'Total'])

    # --- PASO 3: Crear el Gráfico de Torta para "Carne, Pescado y Huevo" ---
    fig_carne = px.pie(
        df_para_grafico,
        names=df_para_grafico.index,  # Las etiquetas de las porciones vienen del índice
        values='Conteo_Carne_Pescado_Huevo',  # Los tamaños de las porciones
        title='Proporción de Consumo de Carne, Pescado y Huevo',
        color_discrete_sequence=PALETA_COLORES
    )
    # Mejoramos la apariencia para que muestre el porcentaje y la etiqueta
    fig_carne.update_traces(textposition='inside', textinfo='percent')
    st.plotly_chart(fig_carne)

    # --- PASO 4: Crear el Gráfico de Torta para "Leche y Derivados" ---
    fig_leche = px.pie(
        df_para_grafico,
        names=df_para_grafico.index,
        values='Conteo_Leche_Derivados',
        title='Proporción de Consumo de Leche y Derivados',
        color_discrete_sequence = PALETA_COLORES
    )
    fig_leche.update_traces(textposition='inside', textinfo='percent')
    st.plotly_chart(fig_leche)

    comentario_alimentacion = textwrap.dedent("""
    En cuanto a la alimentación vemos como el 81% de los estudiantes consumen proteína de origen animal más de 3 veces por semana.
    Los productos lácteos muestran un comportamiento similar, más de 70% de los estudiantes los consumen más de 3 veces a la semana.
    ***Veamos si la alimentación influye en el desempeño de los estudiantes***
    """)

    st.markdown(comentario_alimentacion)
    #Calculamos la media de puntaje global segun alimentacion
    puntaje_medio_carne = icfes_df.groupby(col_carne)[col_puntaje].mean()
    puntaje_medio_leche = icfes_df.groupby(col_leche)[col_puntaje].mean()
    #Agrupamos los datos en un solo df
    df_rendimiento = pd.DataFrame({
        'Puntaje Global Medio (Carne, etc.)': puntaje_medio_carne,
        'Puntaje Global Medio (Leche, etc.)': puntaje_medio_leche
    })
    # Rellenar valores nulos con 0 y redondear a dos decimales
    df_rendimiento = df_rendimiento.fillna(0).round(2)
    #Renombramos el indice para mayor claridad
    df_rendimiento.rename(index={'-': 'Sin Respuesta'}, inplace=True)
    #mostrar la tabla
    st.dataframe(df_rendimiento)

    conclusion_alimenrtacion_rendimiento = textwrap.dedent("""
    Como vemos en la tabla anterior, hay una clara correlación positiva entre el consumo regular de proteína de origen animal y el desempeño de los estudiantes.
    Los estudiantes que consumen carne y lácteos todos los días presentan la media de puntaje más alta, en contraste con los que no consumen estos productos que presentan el promedio más bajo.
    Es así como se puede empezar a esbozar la teoría de que a mayor poder adquisitivo mejor desempeño en las pruebas.
    """)
    st.markdown(conclusion_alimenrtacion_rendimiento)


    conclusion_intro = textwrap.dedent("""**Análisis Descriptivo Preliminar:**

La exploración inicial de los datos muestra que el puntaje global promedio de los estudiantes se sitúa en 266.45.
Hay una fuerte correlación positiva entre la alimentación de los jóvenes y su resultado en las pruebas, para profundizar este análisis vamos a enfocarnos en las calificaciones obtenidas en la materia inglés.

**Foco en la Habilidad de Inglés:**

Un análisis específico de los resultados de inglés arroja un puntaje promedio de 54,35. La desviación estándar, que es de 12,67, sugiere una alta dispersión en las calificaciones. Esto se confirma al observar que los puntajes abarcan todo el espectro posible, desde 0 hasta 100, lo que indica una gran heterogeneidad en el dominio del idioma por parte de los estudiantes. El siguiente paso será analizar la distribución de estos puntajes.
Del análisis anteriormente descrito nos surgen tres preguntas que nos disponemos acontestar:
1. **¿Existe una correlación entre el estrato socioeconómico de un estudiante y su puntaje en la prueba de inglés?**
2. **¿Se observan diferencias significativas en el puntaje global promedio entre los géneros masculino y femenino en los resultados de las pruebas ICFES 2011?**
3. **Análisis del Acceso a Becas y su Impacto en el Rendimiento:**
    1. **¿Qué proporción de los estudiantes evaluados son beneficiarios de programas de becas como "Generación E"?**
    2. **¿Cómo se compara el rendimiento académico (medido por el puntaje global) de los estudiantes becados frente a los no becados?**
""")

    st.markdown(conclusion_intro)

# --- Sección 1: Análisis de los resultados de inglés ---
elif opcion_seleccionada == 'Análisis de Resultados de Inglés':
    st.header('Análisis de los resultados de inglés')

    # --- KPIs o Métricas Clave ---
    # Calculamos las métricas
    puntaje_promedio = icfes_df['PUNT_INGLES'].mean()
    nivel_mas_comun = icfes_df['DESEMP_INGLES'].mode()[0]
    total_estudiantes_ingles = icfes_df['DESEMP_INGLES'].notna().sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Puntaje Promedio", f"{puntaje_promedio:.2f}")
    col2.metric("Nivel Más Común", nivel_mas_comun)
    col3.metric("Total de Estudiantes", f"{total_estudiantes_ingles:,}")

    st.markdown("---")  # Separador visual
    st.subheader('Distribución de Estudiantes por Nivel de Inglés\n')

    # Preparamos los datos
    english_lvl_df = icfes_df['DESEMP_INGLES'].value_counts().reset_index()
    english_lvl_df.columns = ['Nivel_Ingles', 'Total_estudiantes']

    # Definimos el orden correcto de los niveles
    orden_niveles = ['A-', 'A1', 'A2', 'B1', 'B+']

    # Gráfico de barras ORDENADO
    fig_distribucion = px.bar(
        english_lvl_df,
        x='Nivel_Ingles',
        y='Total_estudiantes',
        title='Distribución de Estudiantes por Nivel de Inglés',
        labels={'Nivel_Ingles': 'Nivel de Desempeño', 'Total_estudiantes': 'Cantidad de Estudiantes'},
        color_discrete_sequence=PALETA_COLORES,
        category_orders={'Nivel_Ingles': orden_niveles}  # ¡Esta es la línea clave!
    )
    fig_distribucion.update_layout(xaxis_title="Nivel de Desempeño", yaxis_title="Cantidad de Estudiantes")
    st.plotly_chart(fig_distribucion, use_container_width=True)

    conclusion_ingles = textwrap.dedent("""Vemos como el 85% de los estudiantes estuvieron en los 3 primeros niveles A-, A1, A2. Solo el 15% quedó en los niveles superiores.
Los bajos resultados obtenidos nos llevan a preguntarnos si un mayor estrato socioeconómico se relaciona con un mejor resultado en dicha materia, veamos la distribución socioeconómica de los estudiantes.
                                        """)
    st.markdown(conclusion_ingles)

    # Calculamos la media de puntaje global segun alimentacion
    puntaje_medio_carne = icfes_df.groupby(col_carne)[col_puntaje].mean()
    puntaje_medio_leche = icfes_df.groupby(col_leche)[col_puntaje].mean()
    # Agrupamos los datos en un solo df
    df_rendimiento = pd.DataFrame({
        'Puntaje Global Medio (Carne, etc.)': puntaje_medio_carne,
        'Puntaje Global Medio (Leche, etc.)': puntaje_medio_leche
    })

# --- Sección 2: Análisis socieconómico ---
elif opcion_seleccionada == 'Análisis Socioeconómico':
    st.header('Análisis socieconómico')

    st.subheader('Distribución por estratos')
    estrato_count = icfes_df['FAMI_ESTRATOVIVIENDA'].value_counts()
    estrato_count_df = estrato_count.reset_index()
    estrato_count_df.columns = ['Estrato_familia','Total_estudiantes']

    st.dataframe(estrato_count_df, use_container_width=False)

    # Gráfico de torta de estratos con la nueva paleta de colores
    fig_pie_estrato = px.pie(
        estrato_count_df,
        names='Estrato_familia',
        values='Total_estudiantes',
        title='Proporción de Estudiantes por Estrato',
        color_discrete_sequence=PALETA_COLORES # Aplicamos la paleta aquí
    )
    fig_pie_estrato.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie_estrato)

    pie_estrato=textwrap.dedent("""
    Aproximadamente el 75% de los estudiantes que presentaron la prueba son de estratos 2 y 3.
    Solo el 9% pertenecen a los estratos 4, 5 o 6. Veamos cómo fueron los resultados de inglés según el estrato de los estudiantes.
    """)
    st.markdown(pie_estrato)

    st.subheader('Estudiantes según su nivel de inglés y su estrato')
    crosstab_strata_english = pd.crosstab(icfes_df['FAMI_ESTRATOVIVIENDA'], icfes_df['DESEMP_INGLES'], dropna=False)
    st.dataframe(crosstab_strata_english)


    est_por_ingles_estrato=textwrap.dedent("""
    En la anterior tabla vemos cómo los estudiantes de estratos más altos tuvieron mejores resultados en cuanto a su inglés, para corroborar esto veamos un gráfico de coorelación
            """)
    st.markdown(est_por_ingles_estrato)

    icfes_df['ESTRATO_NUMERICO'] = pd.to_numeric(icfes_df['FAMI_ESTRATOVIVIENDA'], errors='coerce')

    # Eliminar filas donde el estrato o el puntaje no son válidos para un análisis limpio
    df_plot = icfes_df.dropna(subset=['ESTRATO_NUMERICO', 'PUNT_GLOBAL'])


    st.header('Correlación entre Estrato y Puntaje Global')

    # Crear el gráfico de dispersión usando la nueva columna numérica para el eje x
    fig_scatter = px.scatter(
        df_plot,
        x='ESTRATO_NUMERICO',
        y='PUNT_GLOBAL',
        title='Correlación entre Estrato Socioeconómico y Puntaje Global',
        trendline="ols",  # La línea de regresión ahora funcionará
        labels={
            'ESTRATO_NUMERICO': 'Estrato Socioeconómico (Numérico)',
            'PUNT_GLOBAL': 'Puntaje Global'
        },
        color_discrete_sequence=[PALETA_COLORES[1]] # Usando el color dorado de tu paleta
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader('Distribución del Puntaje Global por Estrato')

    # Asegúrate de que df_plot de la sección anterior está disponible
    # (con la columna 'ESTRATO_NUMERICO' y sin NaNs)

    fig_boxplot = px.box(
        df_plot,
        x='ESTRATO_NUMERICO',
        y='PUNT_GLOBAL',
        title='Distribución del Puntaje Global por Estrato Socioeconómico',
        labels={
            'ESTRATO_NUMERICO': 'Estrato Socioeconómico',
            'PUNT_GLOBAL': 'Puntaje Global'
        },
        color='ESTRATO_NUMERICO',  # Colorear cada caja de un color diferente
        color_discrete_sequence=PALETA_COLORES,
        category_orders = {'ESTRATO_NUMERICO': [1, 2, 3, 4, 5, 6]}
    )

    # Esto asegura que los ejes se muestren como categorías discretas (1, 2, 3...)
    fig_boxplot.update_xaxes(type='category')

    st.plotly_chart(fig_boxplot)

    conclusion = textwrap.dedent("""
        **Conclusión:** Los anteriores gráficos muestran una clara tendencia positiva. A medida que aumenta el estrato socioeconómico, la mediana del puntaje global (la línea dentro de cada caja) también tiende a aumentar. Además, la dispersión de los puntajes parece ser diferente entre los estratos.
    """)
    st.markdown(conclusion)

if opcion_seleccionada == 'Análisis de resultados por género':
    st.header('¿Cómo fueron los resultados según el género?')

    st.subheader('Estudiantes por género')

    # Calculate the counts for each gender
    gender_counts = icfes_df['ESTU_GENERO'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Number of Students']

    media_por_genero = icfes_df.groupby('ESTU_GENERO')['PUNT_GLOBAL'].mean()
    media_por_genero = media_por_genero.reset_index()

    # Create the interactive bar plot with specified colors
    gender_distribution = px.bar(gender_counts, x='Gender', y='Number of Students', title='Number of Students by Gender',
                 color='Gender',
                 color_discrete_map={'F': 'pink', 'M': 'steelblue'})  # Assuming default male color is steelblue

    # Show the plot
    st.plotly_chart(gender_distribution)

    comentario_genero = textwrap.dedent(""" 
    El gráfico anterior nos enseña cómo el accesoa a la educación está relativamente parejo aunque se presentaron 5.000 mujeres más al exámen la diferencia es menor al 10%. 
    Hubo 30 estudiantes que no se identificaron con ningún género lo cual se puede entender por el año de los resultados, en esa época no había tanta inclusión.
    
    **Veamos los resultados según el género:**\n
    """)

    st.markdown(comentario_genero)

    st.dataframe(media_por_genero, use_container_width=False)

    fig_violin = px.violin(
        icfes_df,
        x='ESTU_GENERO',
        y='PUNT_GLOBAL',
        color='ESTU_GENERO',
        title='Distribución del Puntaje Global por Género',
        labels={'ESTU_GENERO': 'Género', 'PUNT_GLOBAL': 'Puntaje Global'},
        color_discrete_map={'F': 'pink', 'M': 'lightblue', '-': 'gray'},
        box=True,
        points="all"
    )
    st.plotly_chart(fig_violin)

    conclusion_violin = textwrap.dedent("""
    En este gráfico vemos como la dispersión tanto del género masculino como del femenino es baja como lo indica su rango intercuart[ilico, no hay gran cantidad de valores atípicos.
    La mediana de calificaciones más alta lo tuvieron los hombres seguidos por las personas que no especificaron su género.
    La distribución de datos tanto de hombres como mujeres es muy parecida.
    """)
    st.markdown(conclusion_violin)
    comentario_genero_violin = textwrap.dedent("""
    Por último echemos un vistaso a la **distribución de género por estrato:**\n
    """)
    st.markdown(comentario_genero_violin)

    # Define custom colors for gender categories
    gender_colors = {'-': 'gray', 'F': 'pink', 'M': 'lightblue'}

    # Ensure the normalized strata are treated as discrete for the x-axis order
    strata_order = sorted(
        icfes_df['FAMI_ESTRATOVIVIENDA'].dropna().unique())  # Get unique non-NaN strata and sort them
    # Add NaN to the order if it exists in the data and we want to include it
    if np.nan in icfes_df['FAMI_ESTRATOVIVIENDA'].unique():
        strata_order.append(np.nan)  # Append NaN to the end or handle as desired
    strata_gender_counts = icfes_df.groupby(['FAMI_ESTRATOVIVIENDA', 'ESTU_GENERO']).size().reset_index(
        name='Number of Students')

    # Create a grouped bar plot using Plotly Express
    estrato_por_genero = px.bar(strata_gender_counts,
                 x='FAMI_ESTRATOVIVIENDA',
                 y='Number of Students',
                 color='ESTU_GENERO',  # Color bars by gender
                 barmode='group',  # Create grouped bars
                 title='Estudiantes por estrato socieconómico',
                 color_discrete_map=gender_colors,  # Apply custom colors
                 category_orders={"FAMI_ESTRATOVIVIENDA": strata_order})  # Ensure stratum order on x-axis


    st.plotly_chart(estrato_por_genero)

    conclusion_estudiantes_genero = textwrap.dedent("""
    **Conclusión:** Este último gráfico es muy revelador ya que como habíamos visto en los resultados de inglés, hay una correlación posistiva entre el estrato socioeconómico y el desempeño en las pruebas, 
    así mismo el gráfico de estrato por género nos permite observar cómo en estratos más bajos la población femenina es significativamente mayor, mientras que en los estratos más altos apenas hay diferencias.\n
    Esto explica en cierta medida la diferencia (que además es muy pequeña) entre las medias de lso puntajes globales obtenidos, sin olvidar que estamos en un país muy machista en donde factores como el embarazo adolescente y el trabajo en labores del hogar golpean mucho más fuerte a las mujeres. 
    """)

    st.markdown(conclusion_estudiantes_genero)

# --- Sección 4: Análisis de Estudiantes Becados ('Generación E') ---
elif opcion_seleccionada == 'Análisis de Becados':
    st.header('Análisis de Estudiantes Becados')

    # --- 1. ¿Qué tipos de becas hay y cuántos estudiantes hay? ---
    st.subheader('Tipos de Beca y Cantidad de Estudiantes')

    # Filtrar solo estudiantes que pertenecen a Generación E
    df_becados = icfes_df[icfes_df['ESTU_GENERACION-E'] != 'NO']

    beca_counts = df_becados['ESTU_GENERACION-E'].value_counts().reset_index()
    beca_counts.columns = ['Tipo de Beca', 'Número de Estudiantes']

    # Gráfico de barras para la distribución de becas
    fig_becas_dist = px.bar(
        beca_counts,
        x='Tipo de Beca',
        y='Número de Estudiantes',
        title='Distribución de Estudiantes por Tipo de Beca "Generación E"',
        text_auto=True,  # Muestra los valores en las barras
        color='Tipo de Beca',
        color_discrete_sequence=PALETA_COLORES
    )
    fig_becas_dist.update_layout(xaxis_title="Modalidad de Beca", yaxis_title="Cantidad de Estudiantes")
    st.plotly_chart(fig_becas_dist, use_container_width=True)

    st.markdown("""
    La mayoría de los estudiantes beneficiarios del programa pertenecen a la modalidad **GRATUIDAD**, que apoya a jóvenes en condiciones de vulnerabilidad socioeconómica. Las modalidades de **"EXCELENCIA"** (Nacional y Departamental) son más minoritarias, pues premian a los estudiantes con los más altos puntajes.
    """)
    st.markdown("---")

    # --- 2. ¿Cómo fue el desempeño de los estudiantes becados? ---
    st.subheader('¿Cómo fue el desempeño de los estudiantes becados?')

    # Crear el gráfico de cajas para comparar distribuciones de puntajes
    fig_rendimiento = px.box(
        icfes_df,
        x='ESTU_GENERACION-E',
        y='PUNT_GLOBAL',
        color='ESTU_GENERACION-E',
        title='Distribución de Puntaje Global por Tipo de Beca',
        labels={'ESTU_GENERACION-E': 'Grupo de Estudiante', 'PUNT_GLOBAL': 'Puntaje Global'},
        category_orders={'ESTU_GENERACION-E': ['NO', 'GENERACION E - EQUIPO', 'GENERACION E - EXCELENCIA DEPARTAMENTAL',
                                               'GENERACION E - EXCELENCIA NACIONAL']},
        color_discrete_sequence=PALETA_COLORES
    )
    st.plotly_chart(fig_rendimiento, use_container_width=True)

    st.markdown("""
    Este gráfico es muy revelador. Se observa que los estudiantes con becas de **Excelencia** tienen un rendimiento sobresaliente, con medianas de puntaje significativamente más altas que el resto. Los estudiantes del programa **Gratuidad** tienen un desempeño similar al promedio de los no becados, lo cual es lógico ya que el criterio principal de esta beca es socioeconómico, no de excelencia académica.
    """)
    st.markdown("---")

    # --- 3. ¿A qué estratos pertenecen los estudiantes becados? ---
    st.subheader('¿A qué estratos pertenecen los estudiantes becados?')

    # Contar becados por estrato
    estrato_becados = df_becados['FAMI_ESTRATOVIVIENDA'].value_counts().reset_index()
    estrato_becados.columns = ['Estrato', 'Número de Estudiantes']

    fig_estrato_becados = px.pie(
        estrato_becados,
        names='Estrato',
        values='Número de Estudiantes',
        title='Proporción de Estudiantes Becados por Estrato Socioeconómico',
        color_discrete_sequence=PALETA_COLORES
    )
    fig_estrato_becados.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_estrato_becados, use_container_width=True)

    st.markdown("""
    La gran mayoría de los estudiantes becados (más del 95%) pertenecen a los estratos **1 y 2**. Esto demuestra que el programa "Generación E" está bien focalizado y cumple su objetivo de apoyar a la población con mayores necesidades económicas.
    """)
    st.markdown("---")

    # --- 4. ¿Cómo es la distribución por género de los becados? ---
    st.subheader('¿Cómo es la distribución por género de los becados?')

    genero_becados = df_becados['ESTU_GENERO'].value_counts().reset_index()
    genero_becados.columns = ['Género', 'Número de Estudiantes']

    fig_genero_becados = px.pie(
        genero_becados,
        names='Género',
        values='Número de Estudiantes',
        title='Distribución por Género de Estudiantes Becados',
        color='Género',
        color_discrete_map={'F': '#F7B828', 'M': '#5B97B1'}
    )
    st.plotly_chart(fig_genero_becados, use_container_width=True)

    st.markdown("""
    En la distribución por género de los beneficiarios, se observa una **mayoría de mujeres**. Esto puede reflejar las tendencias demográficas generales en el acceso a la educación superior en el país.
    """)

elif opcion_seleccionada == 'Conclusiones':
    st.header('Conclusiones')

    conclusion1 = textwrap.dedent("""
    La conclusión más clara del análisis es que el estrato socioeconómico tiene un impacto directo en el desempeño de los estudiantes, de los datos analizados podemos deducir que en gran parte esto se debe a un déficit en la alimentación ya que a su vez esta, tiene una fuerte corelación positiva con los resultados de los exámenes.\n
    En cuanto al desempeño en las pruebas de inglés el resultado es un reflejo de lo anteriormente mencionado.\n
    En cuanto a los resultados según el género vemos como las mujeres tienen un desempeño un poco más bajo respecto a,la mediana de los resultados del género opuesto, lo anterior no se puede explicar bajo el entendido de un país desigual en términos de género, en el que las mujeres se ven afectadas por problemas como el embarazo adolescente y trabajos en el hogar que no afectan de igual forma a los hombres.
    Son muy destacables los resultados obtenidos por los estudiantes becados, cuya medaiana de puntaje estuvo muy por encima de la de otros estudiantes.
    """)
    st.markdown(conclusion1)
