import sys
import sklearn
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px


print(f"Python Version: {sys.version}")
print(f"Scikit-learn Version: {sklearn.__version__}")
print(f"Pandas Version: {pd.__version__}")
print("Streamlit version:", st.__version__)
print("Seaborn version:", sns.__version__)

# Load model and preprocessor (if any)
model = pickle.load(open('Churn Prediction.pkl', 'rb'))

# Function to encode categorical features
def encode_features(df):
    label_encoder = LabelEncoder()
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    return df

# Set halaman dengan konfigurasi
st.set_page_config(page_title="Customer Churn Prediction & Analysis Dashboard", layout="wide")

# Mengubah tampilan title halaman
st.markdown(
    """
    <style>
    .title-header {
        text-align: center;                /* Pusatkan teks */
        color:  #b57974 !important;                      /* Warna teks */
        font-family: Arial, sans-serif;    /* Font */
        font-size: 40px;                   /* Ukuran font */
        padding: 15px;                     /* Padding untuk memberi ruang di dalam kotak */
        background-color: #fee5e3;         /* Warna latar belakang */
        border-radius: 10px;               /* Membuat sudut melengkung */
        margin-bottom: 20px;               /* Jarak bawah */
    }
    </style>
    <h1 class="title-header">Customer Churn Prediction & Analysis Dashboard</h1>
    """,
    unsafe_allow_html=True
)
# Menambahkan latar belakang gradien dengan warna lembut
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg,#fefdf9);
        background-size: 400% 400%;
        animation: gradientAnimation 10s ease infinite;
    }
    .stSidebar {
        background-color: #d8a5a1; /* Dusty Rose */
    }
            
     .css-1d391kg { 
        background-color: #d8a5a1; /* Dusty Rose untuk sidebar */
        color: #ffffff;  /* Warna teks sidebar */
    }

    /* Mengubah warna menu aktif */
    .css-1d391kg .active {
        background-color: #f0e0d6;  /* Light Cream untuk menu yang aktif */
    }

    /* Mengubah warna saat hover pada menu */
    .css-1d391kg .st-ae {
        background-color: #f0e0d6; /* Light Cream saat hover */
    }
            
    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
""", unsafe_allow_html=True)


import streamlit as st
from streamlit_option_menu import option_menu

# Sidebar with menu options
with st.sidebar:
    selected = option_menu("Main Menu", ["Predicion", "Dashboard"], 
                           icons=["house", "bar-chart"], 
                           menu_icon="cast", default_index=0, orientation="vertical")

# Function to show the prediction page
def show_prediction_page():
    st.title("Customer Churn Prediction")
    # Input Features - Deskripsi untuk setiap fitur
    st.subheader("Enter Customer Data:")

    # Membagi tampilan menjadi 2 kolom
    col1, col2,col3,col4 = st.columns(4)

    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'], help="Select the customer's gender.")
        senior_citizen = st.selectbox('Senior Citizen?', ['Yes', 'No'], help="Select 'Yes' if the customer is over 65 years old.")
        partner = st.selectbox('Has Partner?', ['Yes', 'No'], help="Select 'Yes' if the customer has a partner.")
        dependents = st.selectbox('Has Dependents?', ['Yes', 'No'], help="Select 'Yes' if the customer has dependents.")
        
    with col2:
        phone_service = st.selectbox('Phone Service?', ['Yes', 'No'], help="Select 'Yes' if the customer has phone service.")
        multiple_lines = st.selectbox('Multiple Lines?', ['Yes', 'No'], help="Select 'Yes' if the customer has multiple phone lines.")
        online_security = st.selectbox('Online Security?', ['Yes', 'No'], help="Select 'Yes' if the customer uses online security services.")
        online_backup = st.selectbox('Online Backup?', ['Yes', 'No'], help="Select 'Yes' if the customer uses online backup services.")
        device_protection = st.selectbox('Device Protection?', ['Yes', 'No'], help="Select 'Yes' if the customer uses device protection services.")
    
    with col3:
        tenure = st.number_input('Tenure (Months)', min_value=0, max_value=100, step=1, help="Enter the number of months the customer has subscribed.")
        internet_service = st.selectbox('Internet Service Type', ['DSL', 'Fiber optic', 'No'], help="Select the type of internet service the customer has.")
        tech_support = st.selectbox('Tech Support?', ['Yes', 'No'], help="Select 'Yes' if the customer uses technical support services.")
        streaming_tv = st.selectbox('Streaming TV?', ['Yes', 'No'], help="Select 'Yes' if the customer uses streaming TV services.")
        streaming_movies = st.selectbox('Streaming Movies?', ['Yes', 'No'], help="Select 'Yes' if the customer uses streaming movie services.")
    with col4: 
        contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'], help="Select the type of contract the customer has.")
        paperless_billing = st.selectbox('Paperless Billing?', ['Yes', 'No'], help="Select 'Yes' if the customer uses paperless billing.")
        payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], help="Select the payment method used by the customer.")
        monthly_charges = st.number_input('Monthly Charges', min_value=0.0, step=1.0, help="Enter the monthly charges paid by the customer.")
        total_charges = st.number_input('Total Charges', min_value=0.0, step=1.0, help="Enter the total charges paid by the customer.")

    # Mengubah data input menjadi format yang sesuai dengan model
    input_data = pd.DataFrame([[gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines,
                                internet_service, online_security, online_backup, device_protection, tech_support,
                                streaming_tv, streaming_movies, contract, paperless_billing, payment_method,
                                monthly_charges, total_charges]],
                              columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                                       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                                       'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])

    # Pastikan urutan fitur input sama dengan yang digunakan untuk pelatihan model
    input_data = input_data[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                             'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                             'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                             'MonthlyCharges', 'TotalCharges']]

    # Encode fitur kategori
    input_data = encode_features(input_data)

    # Fungsi untuk menampilkan hasil prediksi
    def show_prediction(prediction):
        if prediction == 1:
            st.error("⚠️ The customer is predicted to churn.")
        else:
            st.success("🎉 The customer is predicted to stay.")

    # Prediction button
    if st.button('Predict Churn'):
        prediction = model.predict(input_data)
        show_prediction(prediction[0])

    # Additional explanation
    st.write("💡 **What is Churn Prediction?**")
    st.write("Churn prediction helps identify whether a customer is likely to discontinue the service or end their contract. "
            "This model supports businesses in identifying at-risk customers and implementing strategies to retain them.")

# Memuat data baru
data = pd.read_csv('merged_data.csv')

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

def update_all_charts_layout(fig):
    fig.update_layout(
        xaxis=dict(
            title_font=dict(size=16, color='black'),  # Ukuran dan warna font untuk judul sumbu X
            tickfont=dict(size=12, color='black')  # Ukuran dan warna font untuk label sumbu X
        ),
        yaxis=dict(
            title_font=dict(size=16, color='black'),  # Ukuran dan warna font untuk judul sumbu Y
            tickfont=dict(size=12, color='black')  # Ukuran dan warna font untuk label sumbu Y
        ),
        font=dict(size=14, color='black'),  # Ukuran dan warna font untuk elemen lain seperti title dan label
        plot_bgcolor='white'  # Mengatur latar belakang plot menjadi putih
    )
    return fig


# Fungsi untuk menampilkan dashboard dengan insight baru
def show_dashboard_page():
    st.title("Customer Churn Dashboard📊")
    # Add custom CSS for tab styling
    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] {
            font-size: 20px;      /* Change font size */
            font-weight: bold;    /* Make text bold */
            color: black;         /* Change text color */
        }
        .stTabs [data-baseweb="tab"] {
            padding: 20px 40px;   /* Add padding for larger tabs */
            border-radius: 5px;   /* Optional: Add rounded corners */
            background-color: #f0f0f0; /* Background color */
        }
        .stTabs [aria-selected="true"] { 
            background-color: #d8a5a1; /* Highlight selected tab */
            font-weight: bold; 
            color: white; /* Text color for active tab */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Create a tab-like navigation using st.radio
    tab1, tab2, tab3= st.tabs([
        "Service and Rating",
        "Customer Demographics",
        "Customer Status and Churn"
    ])

    with tab2:
        data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')  # Mengubah ke numerik
        data['TotalCharges'] = data['TotalCharges'].fillna(0)  # Mengisi NaN dengan 0

        # Total pelanggan dan pendapatan
        total_customers = data.shape[0]  
        total_revenue = data['TotalCharges'].sum()  

        # Churn Rate
        churn_rate = data['Churn'].mean() * 100  

        # Metrik tambahan
        average_tenure = data['tenure'].mean()  # Rata-rata lama berlangganan
        arpu = total_revenue / total_customers  # Pendapatan rata-rata per pelanggan
       
    
        # Membuat fungsi box HTML dengan warna
        def colored_box(label, value, background):
            return f"""
            <div style="padding: 5px; border-radius:20px; background-color: {background}; text-align: center; color: black; margin: 10px 0;">
                <h3 style="margin: 0;">{label}</h3>
                <p style="margin: 5px 0; font-size: 24px;">{value}</p>
            </div>
            """

        # Tampilkan metrik dengan box berwarna
        col1, col2, col3,col4,col5 = st.columns(5)

        with col1:
            st.markdown(colored_box("Total Customers", f"{total_customers:,}", "#fee5e3"), unsafe_allow_html=True)
        with col2:
            st.markdown(colored_box("Total Revenue", f"${total_revenue:,.2f}", "#fee5e3"), unsafe_allow_html=True)
        with col3:
            st.markdown(colored_box("Churn Rate", f"{churn_rate:.2f}%", "#fee5e3"), unsafe_allow_html=True)
        with col4: 
            st.markdown(colored_box("Average Tenure", f"{average_tenure:.2f} months", "#fee5e3"), unsafe_allow_html=True)
        with col5:
            st.markdown(colored_box("ARPU", f"${arpu:.2f}", "#fee5e3"), unsafe_allow_html=True)
               
        
        
        # Membuat 3 kolom untuk menampilkan 3 visualisasi secara horizontal
        col1, col2,col3 = st.columns(3)

        with col1:
            st.subheader("Customer Gender Distribution")
            # Distribusi Gender (Pie chart)
            gender_dist = px.pie(
                data_frame=data,
                names='gender',
                color='gender',
                color_discrete_map={'Male': '#d8a5a1', 'Female': '#a46f64'}  # Sesuaikan dengan tema warna yang lebih soft dan gelap
            )
            st.plotly_chart(gender_dist)

        with col2:
            st.subheader("SeniorCitizen Customer Distribution")
            # Distribusi SeniorCitizen: Pie chart
            senior_dist = px.pie(
                data_frame=data,
                names='SeniorCitizen',
                color='SeniorCitizen',
                color_discrete_map={0: '#a46f64', 1: '#d8a5a1'}  # Warna lembut dan gelap untuk Senior dan Non-Senior
            )
            st.plotly_chart(senior_dist)

        with col3:
            st.subheader("Distribution Partners and Customer Dependents")
            # Distribusi Partner dan Dependents dalam Bar Chart
            partner_dependents = data.groupby(['Partner', 'Dependents']).size().reset_index(name='Count')

            partner_dependents_bar = px.bar(
                partner_dependents,
                x='Partner',
                y='Count',
                color='Dependents',
                labels={'Count': 'Number of Customers', 'Partner': 'Status Partner', 'Dependents': ' Dependents Status'},
                color_discrete_map={'Yes': '#d8a5a1', 'No': '#a46f64'},  # Kombinasi warna yang lebih lembut
                barmode='group'  # Grouping the bars for each partner status
            )
            partner_dependents_bar = update_all_charts_layout(partner_dependents_bar)
            st.plotly_chart(partner_dependents_bar)

        col1 = st.container()
        with col1:
            st.subheader("Customer Tenure Distribution")
            # Distribusi Tenure: Histogram
            fig_tenure = px.histogram(
                data,
                x='tenure',
                labels={'tenure': 'Subscription Period (Tenure)'},
                color_discrete_sequence=['#d8a5a1'],  # Gunakan warna latar belakang yang lembut untuk histogram
            )
            fig_tenure = update_all_charts_layout(fig_tenure)
            st.plotly_chart(fig_tenure, use_container_width=True)

    with tab1:
        # Membuat 3 kolom untuk menampilkan 3 visualisasi secara horizontal
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Internet Services Rating")
            # Group by 'InternetService' and 'Satisfaction Score' and get the size of each group
            internet_type = data.groupby(by=['InternetService', 'Satisfaction Score']).size().reset_index(name='Count')

            # Replace 'None' with 'Not Subscriber' in 'InternetService'
            internet_type['InternetService'] = internet_type['InternetService'].replace({'None': 'Not Subscriber'})

            # Extract the list of unique internet types
            internet_type_list = internet_type['InternetService'].unique()

            # Initialize an empty dictionary to store percentage values
            percent_dict = {}

            # Calculate the percentage for each satisfaction score and internet type
            for num in range(1, 6):
                percent_list = []
                for i_type in internet_type_list:
                    # Get the count for the specific satisfaction score and internet type
                    count_var = internet_type[(internet_type['Satisfaction Score'] == num) & (internet_type['InternetService'] == i_type)]['Count'].values[0]
                    
                    # Calculate the sum for the internet type to calculate percentage
                    sum_var = internet_type[internet_type['InternetService'] == i_type]['Count'].sum()
                    
                    # Calculate percentage
                    percent = round((count_var / sum_var) * 100, 1)
                    percent_list.append(percent)
                
                # Store the percentage values for each satisfaction score
                percent_dict[num] = percent_list

            # Define new color combination for each satisfaction score
            color_rating = ['#d8a5a1', '#f5c6c2', '#a46f64', '#d8a5a1', '#f5c6c2']

            # Function to get counts for a specific satisfaction score
            def internet_viz(score):
                return internet_type[internet_type['Satisfaction Score'] == score]['Count'].to_list()

            # Create the bar chart using Plotly
            fig_internet = go.Figure(data=[
                go.Bar(
                    name=f'{i}',
                    x=internet_type_list,
                    y=internet_viz(i),
                    text=[f'{i} %' for i in percent_dict[i]],
                    marker_color=color
                ) for i, color in zip(range(1, 6), color_rating)
            ])

            # Update layout and appearance of the figure
            fig_internet.update_layout(
                barmode='group', 
                autosize=False, 
                width=1200, 
                height=500,
                margin=dict(l=0, r=0, b=0, t=10, pad=0), 
                plot_bgcolor='white',
                yaxis={'categoryorder': 'total ascending'},
                legend=dict(title='Customer Rating: ', yanchor="top", y=0.98, xanchor="left", x=0.03)
            )

            fig_internet.update_traces(
                textfont_size=12, 
                textangle=0, 
                cliponaxis=False, 
                textposition="outside", 
                textfont_color='black'
            )

            # Display the plot
            fig_internet = update_all_charts_layout(fig_internet)
            st.plotly_chart(fig_internet)

        with col2:
            st.subheader("Phone Services Rating")
            bucket = []
            for i, j in zip(data['PhoneService'], data['MultipleLines']):
                if (i == 'Yes') & (j == 'No'):
                    bucket.append('Single Line')
                elif (i == 'Yes') & (j == 'Yes'):
                    bucket.append('Multi Lines')
                else:
                    bucket.append('Not Subscriber')
            data['PhoneService2'] = bucket
            # Group by 'Phone Service' and 'Satisfaction Score' and get the size of each group
            ps = data.groupby(by=['PhoneService2', 'Satisfaction Score']).size().reset_index(name='Count')
            # Replace 'None' with 'Not Subscriber' in 'Phone Service' (if necessary)
            ps['Phone Service'] = ps['PhoneService2'].replace({'None': 'Not Subscriber'})
            # Extract the list of unique phone service types
            phone_service_list = ps['PhoneService2'].unique()
            # Initialize an empty dictionary to store percentage values
            percent_dict = {}
            # Calculate the percentage for each satisfaction score and phone service
            for num in range(1, 6):
                percent_list = []
                for i_type in phone_service_list:
                    # Get the count for the specific satisfaction score and phone service
                    count_var = ps[(ps['Satisfaction Score'] == num) & (ps['PhoneService2'] == i_type)]['Count'].values[0]
                                        # Calculate the sum for the phone service to calculate percentage
                    sum_var = ps[ps['PhoneService2'] == i_type]['Count'].sum()
                                        # Calculate percentage
                    percent = round((count_var / sum_var) * 100, 1)
                    percent_list.append(percent)
                                    # Store the percentage values for each satisfaction score
                percent_dict[num] = percent_list
            # Define new color combination for each satisfaction score
            color_rating = ['#d8a5a1', '#f5c6c2', '#a46f64', '#d8a5a1', '#f5c6c2']
            # Function to get counts for a specific satisfaction score
            def phone_service_viz(score):
                return ps[ps['Satisfaction Score'] == score]['Count'].to_list()
            # Create the bar chart using Plotly
            fig_phone_service = go.Figure(data=[
                go.Bar(
                    name=f'{i}',
                    x=phone_service_list,
                    y=phone_service_viz(i),
                    text=[f'{i} %' for i in percent_dict[i]],
                    marker_color=color
                ) for i, color in zip(range(1, 6), color_rating)
            ])
            # Update layout and appearance of the figure
            fig_phone_service.update_layout(
                barmode='group', 
                autosize=False, 
                width=1200, 
                height=500,
                margin=dict(l=0, r=0, b=0, t=10, pad=0), 
                plot_bgcolor='white',
                yaxis={'categoryorder': 'total ascending'},
                legend=dict(title='Customer Rating: ', yanchor="top", y=0.98, xanchor="left", x=0.03)
            )
            fig_phone_service.update_traces(
                textfont_size=12, 
                textangle=0, 
                cliponaxis=False, 
                textposition="outside", 
                textfont_color='black'
            )
            # Display the plot
            fig_phone_service = update_all_charts_layout(fig_phone_service)
            st.plotly_chart(fig_phone_service)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribution of Services Used")

            # Modify the column names as per your dataset
            services_used = data[['PhoneService', 'InternetService', 'OnlineSecurity']].apply(pd.Series.value_counts).T

            # Plot pie chart for each service category
            fig_services = px.pie(
                services_used,
                names=services_used.index,
                values=services_used['Yes'],  # Assumes 'Yes' represents usage
                color_discrete_sequence=['#d8a5a1', '#f5c6c2', '#a46f64']  # New color combination
            )

            # Update layout to adjust appearance
            fig_services.update_layout(
                plot_bgcolor='white',
                height=500,
                width=800,
                font=dict(size=14),
            )

            fig_services = update_all_charts_layout(fig_services)
            st.plotly_chart(fig_services)

        with col2: 
            st.subheader("Churn Service Type")
            data_clean = data.dropna(subset=['PhoneService', 'InternetService', 'OnlineSecurity', 'Churn'])
            # Misalnya, ganti 'Yes' dan 'No' dengan 1 dan 0
            data_clean['PhoneService'] = data_clean['PhoneService'].replace({'Yes': 1, 'No': 0})
            data_clean['InternetService'] = data_clean['InternetService'].replace({'Yes': 1, 'No': 0})
            data_clean['OnlineSecurity'] = data_clean['OnlineSecurity'].replace({'Yes': 1, 'No': 0})
            data_clean['Churn'] = data_clean['Churn'].map({True: 1, False: 0, 'Yes': 1, 'No': 0})
            
            # Create a function to calculate churn based on each service
            def churn_by_service(service_column):
                churn_data = data_clean.groupby([service_column, 'Churn']).size().reset_index(name='Count')
                churn_data['Churn'] = churn_data['Churn'].map({1: 'Churned', 0: 'Not Churned'})  # Assuming 'Churn' is coded as 1/0
                return churn_data

            # Calculate churn data for each service
            churn_phone_service = churn_by_service('PhoneService')
            churn_internet_service = churn_by_service('InternetService')
            churn_online_security = churn_by_service('OnlineSecurity')

            # Combine all service data into a single DataFrame with a new column indicating the service type
            churn_phone_service['ServiceType'] = 'Phone Service'
            churn_internet_service['ServiceType'] = 'Internet Service'
            churn_online_security['ServiceType'] = 'Online Security'

            # Combine the DataFrames
            churn_all_services = pd.concat([churn_phone_service, churn_internet_service, churn_online_security], ignore_index=True)

            # Create a bar chart showing churn based on service
            fig_churn_services = px.bar(
                churn_all_services,
                x='ServiceType',  # Show different services on the x-axis
                y='Count',
                color='Churn',
                color_discrete_sequence=['#a46f64', '#d8a5a1'],  # Customize colors for churned and non-churned
                labels={'Churn': 'Churn Status', 'Count': 'Number of Customers'},
                barmode='group'  # Group bars for each service
            )

            # Update layout for better readability
            fig_churn_services.update_layout(
                plot_bgcolor='white',
                height=500,
                width=800,
                font=dict(size=14),
            )

            # Display the chart
            fig_churn_services = update_all_charts_layout(fig_churn_services)
            st.plotly_chart(fig_churn_services, use_container_width=True)


    with tab3:
        customer_status_counts = data_clean['Customer Status'].value_counts()
        status_percentage = (customer_status_counts / len(data_clean)) * 100
        fig_customer_status_bar = px.bar(
            customer_status_counts,
            y=customer_status_counts.index,
            x=customer_status_counts.values,
            labels={"y": "Customer Status", "x": "Number of Customers"},
            color=customer_status_counts.index,
            color_discrete_sequence=["#d8a5a1", "#f5c6c2", "#a46f64"],
            text=customer_status_counts.values  # Menampilkan jumlah pelanggan di setiap bar
        )

        fig_customer_status_bar.update_traces(
            texttemplate='%{text} Customers\n',  # Format persentase
            textposition='outside',  # Menampilkan teks di luar bar
        )
        revenue_by_status = data_clean.groupby('Customer Status')['Total Revenue'].sum()

        fig_revenue_by_status = px.bar(
            revenue_by_status,
            y=revenue_by_status.index,
            x=revenue_by_status.values,
            labels={"y": "Customer Status", "x": "Total Revenue"},
            color=revenue_by_status.index,
            color_discrete_sequence=["#d8a5a1", "#f5c6c2", "#a46f64"],
            text=revenue_by_status.values  # Menampilkan jumlah revenue di setiap bar
        )

        fig_revenue_by_status.update_traces(
            texttemplate='$%{text:.2s}',  # Format revenue dengan simbol dolar
            textposition='outside',  # Menampilkan teks di luar bar
        )
        churned_data = data_clean[data_clean['Customer Status'] == 'Churned']
        churned_tenure_counts = churned_data['tenure'].value_counts().sort_index()

        churned_tenure_percentage = (churned_tenure_counts / churned_tenure_counts.sum()) * 100
        tenure_categories = pd.cut(churned_data['tenure'], bins=[0, 12, 24, float('inf')],
                                labels=['0-12 month', '13-24 month', '> 24 month'])
        tenure_category_counts = tenure_categories.value_counts().sort_index()
        tenure_category_percentage = (tenure_category_counts / tenure_category_counts.sum()) * 100
        text_labels = [
            f"{percent:.2f}%"
            for value, percent in zip(tenure_category_counts.values, tenure_category_percentage.values)
        ]
        
        fig_churned_tenure = px.bar(
            x=tenure_category_counts.index,
            y=tenure_category_counts,
            labels={"x": "Tenure (month)", "y": "Number of Customers"},
            color=tenure_category_counts.index,
            color_discrete_sequence=['#d8a5a1', '#f5c6c2', '#a46f64'],  # Warna sesuai dengan tema
            text=text_labels,  # Label gabungan jumlah dan persentase
        )
        
        fig_churned_tenure.update_traces(textposition='outside')
        fig_churned_tenure.update_layout(
            plot_bgcolor='white',
            height=500,
            width=800,
            font=dict(size=14),
        )

        churned_data = data_clean[data_clean['Customer Status'] == 'Churned']
        churned_city_counts = churned_data['City'].value_counts()
        top_5_churned_cities = churned_city_counts.head(5)
        top_5_churned_percentage = (top_5_churned_cities / top_5_churned_cities.sum()) * 100
        text_labels = [
            f"{percent:.2f}%"
            for value, percent in zip(top_5_churned_cities.values, top_5_churned_percentage.values)
        ]
        fig_churned_city = px.bar(
            x=top_5_churned_cities.values, 
            y=top_5_churned_cities.index,
            labels={"x": "Number of Churned Customers", "y": "City"},
            color=top_5_churned_cities.index,
            color_discrete_sequence=['#d8a5a1', '#f5c6c2', '#a46f64'],  # Warna sesuai tema
            text=text_labels,  # Menampilkan jumlah dan persentase churned di setiap kota
            orientation='h'  # Membuat bar chart horizontal
        )
        fig_churned_city.update_traces(textposition='outside', )
        fig_churned_city.update_layout(
            plot_bgcolor='white',
            height=500,
            width=800,
            font=dict(size=14),
        )
        
        col1, col2,col3 = st.columns(3)
        with col1:
            st.subheader("Customer Status Distribution")
            fig_customer_status_bar = update_all_charts_layout(fig_customer_status_bar)
            st.plotly_chart(fig_customer_status_bar, use_container_width=True)
        with col2:
            st.subheader("Total Revenue by Customer Status")
            fig_revenue_by_status = update_all_charts_layout(fig_revenue_by_status)
            st.plotly_chart(fig_revenue_by_status, use_container_width=True)
        with col3:
            st.subheader("Churned Customer Retention Duration")
            fig_churned_tenure = update_all_charts_layout(fig_churned_tenure)
            st.plotly_chart(fig_churned_tenure, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Churn Rate by Category")
            # Group data by churn category and count
            churn_category_counts = data['Churn Category'].value_counts().reset_index()
            churn_category_counts.columns = ['Churn Category', 'Count']
            # Sort by Count in ascending order to display categories from smallest to largest
            churn_category_counts = churn_category_counts.sort_values(by='Count', ascending=True)
            # Calculate percentage
            churn_category_counts['Percentage'] = (churn_category_counts['Count'] / churn_category_counts['Count'].sum()) * 100
            # Interactive bar chart
            fig_churn_category = px.bar(
                churn_category_counts,
                x='Count',
                y='Churn Category',
                orientation='h',
                text=churn_category_counts['Percentage'].apply(lambda x: f'{x:.1f}%'),
                labels={'Count': 'Number of Customers', 'Churn Category': 'Category'},
                color_discrete_sequence=['#d8a5a1']  # New color for the bars
            )
            # Update layout
            fig_churn_category.update_layout(
                plot_bgcolor='white',
                height=500,
                width=1000,
                font=dict(size=14),
                xaxis=dict(title_font=dict(size=16), tickfont=dict(size=12)),
                yaxis=dict(title_font=dict(size=16), tickfont=dict(size=12))
            )
            fig_churn_category.update_traces(textfont_size=12, textposition="outside")
            fig_churn_category = update_all_charts_layout(fig_churn_category)
            st.plotly_chart(fig_churn_category)

        with col2:
            st.subheader("Impact of Cities on Churn (Top 5 Cities)")
            fig_churned_city = update_all_charts_layout(fig_churned_city)
            st.plotly_chart(fig_churned_city, use_container_width=True)

        col1 = st.container()

        with col1:
            st.subheader("Churn Rate by Reason")
            # Group data by churn reason and count the number of customers
            churn_reason_counts = data['Churn Reason'].value_counts().reset_index()
            churn_reason_counts.columns = ['Churn Reason', 'Count']
            
            # Sort by Count in descending order to display reasons from highest to lowest
            churn_reason_counts = churn_reason_counts.sort_values(by='Count', ascending=True)
            
            # Calculate the percentage for each reason
            churn_reason_counts['Percentage'] = (churn_reason_counts['Count'] / churn_reason_counts['Count'].sum()) * 100
            
            # Create interactive bar chart with uniform colors
            fig_churn_reason = px.bar(
                churn_reason_counts,
                x='Count',
                y='Churn Reason',
                orientation='h',  # Horizontal bar chart
                text=churn_reason_counts['Percentage'].apply(lambda x: f'{x:.1f}%'),  # Display percentage in bars
                labels={'Count': 'Number of Customers', 'Churn Reason': 'Reason for Churn'},
                color_discrete_sequence=['#d8a5a1']  # New color for the bars
            )
            
            # Customize layout and font sizes
            fig_churn_reason.update_layout(
                plot_bgcolor='white',
                xaxis_title='Number of Customers',
                yaxis_title='Reason for Churn',
                height=800,
                width=1200,  # Adjusted width for better readability
                margin=dict(l=100, r=20, t=40, b=40),  # Add margins
                font=dict(
                    family="Arial, sans-serif",  # Font family
                    size=18,  # Default font size for labels and titles
                    color="#3A3A3A"  # Font color
                ),
                xaxis=dict(
                    title=dict(font=dict(size=16)),  # X-axis title font size
                    tickfont=dict(size=12)  # X-axis tick label font size
                ),
                yaxis=dict(
                    title=dict(font=dict(size=20)),  # Y-axis title font size
                    tickfont=dict(size=12)  # Y-axis tick label font size
                ),
            )
            
            # Customize percentage labels in bars
            fig_churn_reason.update_traces(
                textfont_size=12,  # Font size for percentage labels
                textposition="outside"  # Position outside the bars
            )
            
            # Display the chart in Streamlit
            fig_churn_reason = update_all_charts_layout(fig_churn_reason)
            st.plotly_chart(fig_churn_reason)

        
   

# Main interface logic
if selected == "Predicion":
    show_prediction_page()
elif selected == "Dashboard":
    show_dashboard_page()
