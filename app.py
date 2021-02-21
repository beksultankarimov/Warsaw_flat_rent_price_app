import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost

###############################################################################################

raw_df = pd.read_csv('data_clean.csv')
df = raw_df.drop(['gross_price', ], axis=1)
###############################################################################################
st.write("""# Warsaw flat rental price prediction app""")

st.image('flat2.jpg', use_column_width=True)

if st.checkbox("Show how to use the app/More info", False):
    st.write('''This application is based on data collected from http://www.otodom.pl. Numerous features were selected like area, number of rooms,
     distrit and so on. This app is just proof of the concept and needs a lot of optimization to be a real tool, 
     but the model used for prediction(XGBoost) is legit and predictions are accurate. For more info visit the GitHub repository of this project by 
     the link provided at the bottom.''')
    st.write("""App has two modes: 

    1.Predict the price of the flat by given features
    2.Predict the average flat and it's features by the given budget""")
    st.write("""<-- Change the mode in sidebar on the left""")
st.write('___________________________________________________________________________________________________________________________')
st.sidebar.header('User Input Features')

decision_box = st.sidebar.selectbox('Select the mode:',('Predict the flat by your budget','Predict the price of flat'))
if decision_box == 'Predict the price of flat':
    #def user_input_features():
    district = st.sidebar.selectbox('District',('Bemowo','Białołęka','Bielany', 'Centrum', 'Mokotów', 'Ochota',
        'Praga-Południe', 'Praga-Północ', 'Rembertów', 'Targówek', 'Ursus', 'Ursynów','Wawer','Wesoła','Wilanów', 
        'Wola', 'Włochy','Śródmieście', 'Żoliborz'))
    build_type = st.sidebar.selectbox('Building Type',('Apartment_high_q(apartamentowiec)','Apartment_medium_q(blok)','Private_house_1_fam(dom wolnostojący)', 'Tenement(kamienica)', 'Loft/attic(loft)',
        'Infill(plomba)', 'Private_house_1+_fam(szeregowiec)', 'Loft/attic(loft)'))
    build_mat = st.sidebar.selectbox('Building Material',('Concreate(beton)','Autoclaved_aerated_concrete(beton_komórkowy)','Brick(cegła)', 'Wood(drewno)','Other(inne)', 'Silicate brick(silikat)'
        'Expanded_clay(keramzyt)', 'Concrete_masonry_unit(pustak)', 'Conreate_slab(wielka_płyta)', 'Reinforced_concrete(żelbet)'))
    windows = st.sidebar.selectbox('Windows',('Aluminum(aluminiowe)','Wooden(drewniane)','Plastic(plastikowe)'))
    heating = st.sidebar.selectbox('Heating',('Electric(elektryczne)','Gas(gazowe)','Other(inne)', 'Boiler(kotłownia)', 'Central(miejskie)'))
    status = st.sidebar.selectbox('Status',('Not_ready_yet(do_wykończenia)','Ready(do_zamieszkania)','Renovation(do remontu)'))
    area = st.sidebar.slider('Area', 0, 250, 45 )
    room_num = st.sidebar.slider('Number of Rooms', 1,10, 2)
    floor = st.sidebar.slider('Floor', 0, 30, 3)
    total_floor = st.sidebar.slider('Total Number of Floors', 0, 30, 5)
    year_built = st.sidebar.slider('Year Built', 1890,2021, 2000)
    agd = st.sidebar.multiselect("Equipment", ('dish_washer(zmywarka)','fridge(lodówka)','furniture(meble)','oven(piekarnik)','stove(kuchenka)','tv_set(telewizor)','washer(pralka)'))
    security = st.sidebar.multiselect("Security", ('secure_doors/windows(drzwi/okna_antywłamaniowe)','anti-burglary blinds(rolety antywłamaniowe)','intercom/videophone(domofon/wideofon)','monitoring/security(monitoring/ochrona)',
        'closed_area(teren_zamknięty)','alarm system(system alarmowy)'))
    additional_features = st.sidebar.multiselect("Additional Features", ('balcony(balkon)','basement(piwnica)','garage/parking_space(garaż/miejsce_parkingowe)',
        'only_for_non-smokers(tylko_dla_niepalących)','elevator(winda)', 'separate kitchen(oddzielna kuchnia)', 'utility room(pom. użytkowe)', 'terrace(taras)', 'two-level(dwupoziomowe)', 'garden(ogródek)'
        'available for students(wynajmę również studentom)',))
    media =  st.sidebar.multiselect("Media", ( 'air conditioning(klimatyzacja)', 'telephone(telefon)', 'cable TV(telewizja kablowa)', 'internet'))

    data = {'district': district,
            'build_type': build_type,
            'build_mat': build_mat,
            'windows': windows,
            'heating': heating,
            'status': status,
            'area' : area,
            'room_num':room_num,
            'floor':floor,
            'total_floor':total_floor,
            'year_built':year_built,
            'agd':str(agd),
            'security':str(security),
            'additional_features': str(additional_features),
            'media':str(media)}


    #Copying the input format
    
    input_df = df.loc[df.index ==0]
    for i in input_df.columns:
        input_df[i] = 0

    #declaring input values
    input_df['area'] = area
    input_df['room_num'] = room_num
    input_df['floor'] = floor
    input_df['total_floor'] = total_floor
    input_df['year_built'] = year_built

    add_features = district,build_type,build_mat,windows,heating,status
    add_features_lst = []
    for i in add_features:
        add_features_lst.append(i)

    add_features2 = agd,security,additional_features, media
    for i in add_features2:
        for j in i:
            add_features_lst.append(j)


    for i in input_df.columns:
        for j in add_features_lst:
            if j in i:
                input_df[i] = 1



    # Displays the user input features
    st.subheader('User Input features')
    st.write(input_df[['area','room_num', 'floor', 'total_floor', 'year_built']])
    if st.checkbox("Show all the input features", False):
                st.write(input_df)

    st.write('___________________________________________________________________________________________________________________________')


    # Reads in saved classification model
    load_model = pickle.load(open('WF_model.pkl', 'rb'))

    # Apply model to make predictions
    prediction = int(load_model.predict(input_df))
    st.write('''# Calculated rental price =  ''',prediction, '''PLN''')
    st.write("Please keep in mind that flats require additional deposit equal to 100% of rental fee. So in total you need: ", prediction*2, 'PLN to rent a flat')
else:
    #District option for multiselect
    correct_dist = ['district_ Włochy', 'district_ Mokotów', 'district_ Bielany', 'district_ Targówek', 'district_ Ochota',
        'district_ Żoliborz', 'district_ Ursynów', 'district_ Wola', 'district_ Wawer', 'district_ Białołęka', 'district_ Wilanów',
        'district_ Ursus', 'district_ Praga-Południe', 'district_ Bemowo', 'district_ Centrum', 'district_ Praga-Północ',
        'district_ Śródmieście', 'district_ Rembertów', 'district_ Wesoła']

    #user input
    st.sidebar.write('Budget')
    budget = st.sidebar.slider('Max affordable price of flat', 0, 7000, 2000 )
    st.sidebar.write('Number of Rooms')
    room_wanted = st.sidebar.slider('Max is 4 since dataset do not contain enough flats with more than 4 rooms', 1,4, 2)
    st.sidebar.write('Choose the district')
    district_wanted = st.sidebar.selectbox('Note, that not all price and rooms range might be available in each district ', correct_dist)

    #creating subset filtered by user input
    needed_df = raw_df.loc[(raw_df.gross_price <= budget)&(raw_df.room_num >= room_wanted) ]

    #creating dict with the mean values of the flat features for each district subset
    districts = [' Włochy', ' Mokotów', ' Bielany', ' Targówek', ' Ochota',
       ' Żoliborz', ' Ursynów', ' Wola', ' Wawer', ' Białołęka',
       ' Wilanów', ' Ursus', ' Praga-Południe', ' Bemowo', ' Centrum',
       ' Praga-Północ', ' Śródmieście', ' Warszawa', ' mazowieckie',
       ' Rembertów', ' Wesoła', ' Metro Wilanowska']
    new_districts = []
    for i in districts:
        i = 'district_'+i
        new_districts.append(i)

    districts_dict = dict()
    for i in new_districts:
        districts_dict[i] = needed_df.loc[needed_df[i] == 1]

    def result_dist(district):
        result_df = needed_df.head(1)
        for i in districts_dict[district].columns:
            result_df[i] = districts_dict[district][i].mean()
        return result_df
    #creating the final result df, where all districts are collected together
    final_result = pd.DataFrame(columns=needed_df.columns)
    for j in new_districts:
        final_result = final_result.append(result_dist(j), ignore_index=True)

    #dropping nan(if exists), rounding the probablities of the each feature(important not to convert to int)
    final_result.dropna(inplace=True)
    for i in final_result.columns:
        final_result[i] = round(final_result[i],0)
    
    
    # creating a dictionary that will keep only existing features(the ones without 0)
    result_df_dict = dict()
    for i in correct_dist:
        result_df_dict[i] = final_result.loc[final_result[i] == 1]

    def best_offer(district):
        best_offer = dict()
        for i in result_df_dict[district].columns:
            if result_df_dict[district][i].all() != 0:
                best_offer[i] = result_df_dict[district][i]
        return best_offer
    
    #creating a dict with separeted district average offer
    offer_by_dist = dict()
    for i in correct_dist:
        offer_by_dist[i] = best_offer(i)

    for j in offer_by_dist:
        if j in district_wanted:  
            offer_dict = dict()
            try:
                offer_dict['gross_price'] = str(int(offer_by_dist[j]['gross_price']))+' PLN'
                offer_dict['gross_price'] = str(int(offer_by_dist[j]['gross_price']))+' PLN'
                offer_dict['area'] = str(int(offer_by_dist[j]['area'])) + ' m²'
                offer_dict['room_num'] = str(int(offer_by_dist[j]['room_num'])) + '-room'
                offer_dict['floor'] = int(offer_by_dist[j]['floor'])
                if offer_dict['floor'] == 1:
                    offer_dict['floor'] = str(offer_dict['floor']) + 'st'
                elif offer_dict['floor'] == 2:
                    offer_dict['floor'] = str(offer_dict['floor']) + 'nd'
                else:
                    offer_dict['floor'] = str(offer_dict['floor']) + 'th'
                offer_dict['bulding_height'] = str(int(offer_by_dist[j]['total_floor']))+ ' floors'
                offer_dict['year_built'] = int(offer_by_dist[j]['year_built'])
                          
                            

                num_features = ['area','year_built','room_num','total_floor','gross_price','floor','district_ Włochy', 'district_ Mokotów', 'district_ Bielany', 'district_ Targówek', 'district_ Ochota',
                    'district_ Żoliborz', 'district_ Ursynów', 'district_ Wola', 'district_ Wawer', 'district_ Białołęka', 'district_ Wilanów',
                    'district_ Ursus', 'district_ Praga-Południe', 'district_ Bemowo', 'district_ Centrum', 'district_ Praga-Północ',
                    'district_ Śródmieście', 'district_ Rembertów', 'district_ Wesoła']
                categorical_features = []
                for i in offer_by_dist[j]:
                    if i in num_features:
                        pass
                    else:
                        categorical_features.append(i)
                
                print_result = pd.DataFrame(offer_dict, index =[0])


                st.subheader(j)
                st.write(print_result)
                if st.checkbox("Show the rest of the features", False):
                    st.write(categorical_features)
            except:
                st.write('''# There is no flat with given input :(''')
            
        else: pass
#visually dividing section
st.write('___________________________________________________________________________________________________')
#footer
st.write('Prediction was done based on **XGBoost** model with **99%** accuracy. Data obtained from otodom.pl.')
st.write('Source code:','''[Kaggle](https://www.linkedin.com/in/beksultan-karimov-6a4296179/)''',' or ',
    '''[GitHub](https://www.linkedin.com/in/beksultan-karimov-6a4296179/)''')
st.write('Dataset is available on ', '''[Kaggle](https://www.linkedin.com/in/beksultan-karimov-6a4296179/)''',' and on ',
    '''[GitHub](https://www.linkedin.com/in/beksultan-karimov-6a4296179/)''')
st.write('This project was done by: **Beksultan Karimov**')
st.write('''\n[LinkedIn](https://www.linkedin.com/in/beksultan-karimov-6a4296179/)''')
st.write('''[Facebook](https://www.facebook.com/profile.php?id=100009130718211)''')
st.write('''[GitHub](https://github.com/beksultankarimov)''')


    

