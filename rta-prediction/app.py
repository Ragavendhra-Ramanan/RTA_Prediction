import numpy as np
import joblib
import streamlit as st
from catboost import CatBoostClassifier
from prediction import get_encoded_value, get_prediction

model=joblib.load(r'rta-prediction/Model/catboost.joblib')
encodings=joblib.load(r'rta-prediction/Model/label_encodings')

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")
                   
casualty_fitness=['Normal','Blind','Deaf','Other']
day_of_week=['Friday','Wednesday','Thursday','Tuesday','Saturday','Monday','Sunday']
age_band_driver=['18-30','Under 18','31-50','Over 51',]
sex_of_driver=['Male','Female']
driver_education=['Junior high school','Elementary school','High school','Above high school','Writing & reading','Illiterate']
vehicle_driver_relation=['Employee','Owner','Other']
driving_experience=['5-10yr','2-5yr','Above 10yr','1-2yr','Below 1yr','No Licence']
type_of_vehicle=['Automobile','Medium lorry','Other','Pick up upto 10Q','small public','Stationwagen','small lorry','medium public','large public','Long lorry','Taxi','Motorcycle','Special vehicle','Ridden horse','Turbo','Bajaj','Bicycle']
owner_of_vehicle=['Owner','Governmental','Organization','Other']
service_year_of_vehicle=['2-5yrs','Above 10yr','5-10yrs','1-2yr','Below 1yr','Unknown']
area_accident_occured=['Church areas','Hospital areas','Industrial areas','Market areas','Office areas','Other','Outside rural areas','Recreational areas', 'Residential areas','Rural village areas','School areas']
lanes_or_medians=['Double carriageway (median)','One way','Two-way (divided with broken lines road marking)', 'Two-way (divided with solid lines road marking)','Undivided Two way','other']
road_allignment= ['Escarpments','Gentle horizontal curve','Sharp reverse curve','Steep grade downward with mountainous terrain','Steep grade upward with mountainous terrain','Tangent road with flat terrain','Tangent road with mild grade and flat terrain','Tangent road with mountainous terrain and','Tangent road with rolling terrain']
types_of_junction= ['Crossing','No junction','O Shape','Other','T Shape','X Shape','Y Shape']
road_surface_type=['Asphalt roads','Asphalt roads with some distress','Earth roads','Gravel roads','Other']
road_surface_conditions= ['Dry','Flood over 3cm. deep','Snow','Wet or damp']
light_conditions= ['Darkness - lights lit','Darkness - lights unlit','Darkness - no lighting','Daylight']
weather_conditions= ['Cloudy','Fog or mist','Normal','Other','Raining','Raining and Windy','Snow','Windy']
type_of_collision=['Collision with animals','Collision with pedestrians','Collision with roadside objects','Collision with roadside-parked vehicles','Fall from vehicles','Other','Rollover','Vehicle with vehicle collision','With Train']
vehicle_movement= ['Entering a junction','Getting off','Going straight','Other','Overtaking','Parked','Reversing','Stopping','Turnover','U-Turn','Waiting to go']
casualty_class= ['Driver or rider','Passenger','Pedestrian']
sex_of_casualty= ['Female', 'Male']
age_band_of_casualty=['18-30','31-50','Over 51','Under 18']
work_of_casuality=['Driver','Employee','Other','Self-employed','Student','Unemployed']
fitness_of_casuality=['Blind','Deaf','Normal','Other']
pedestrian_movement=["Crossing from driver's nearside",'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle','Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle','In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)','In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle','Not a Pedestrian','Unknown or other','Walking along in carriageway, back to traffic','Walking along in carriageway, facing traffic']
cause_of_accident=['Changing lane to the left','Changing lane to the right','Driving at high speed','Driving carelessly','Driving to the left','Driving under the influence of drugs','Drunk driving','Getting off the vehicle improperly','Improper parking','Moving Backward','No distancing','No priority to pedestrian','No priority to vehicle','Other','Overloading','Overspeed','Overtaking','Overturning','Turnover']

features=['Time', 'Day_of_week', 'Age_band_of_driver', 'Sex_of_driver',
       'Educational_level', 'Vehicle_driver_relation', 'Driving_experience',
       'Type_of_vehicle', 'Owner_of_vehicle', 'Service_year_of_vehicle',
       'Area_accident_occured', 'Lanes_or_Medians', 'Road_allignment',
       'Types_of_Junction', 'Road_surface_type', 'Road_surface_conditions',
       'Light_conditions', 'Weather_conditions', 'Type_of_collision',
       'Number_of_vehicles_involved', 'Number_of_casualties',
       'Vehicle_movement', 'Casualty_class', 'Sex_of_casualty',
       'Age_band_of_casualty', 'Casualty_severity', 'Work_of_casuality',
       'Fitness_of_casuality', 'Pedestrian_movement', 'Cause_of_accident']

st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)

def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        f_hour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
        f_day_of_week = st.selectbox("Select Day of the Week: ", options=day_of_week)
        f_casualty_fitness = st.selectbox("Select the fitness of the casualty: ", options=casualty_fitness)
        f_age_band_driver=st.selectbox("Select the Age Group of Driver: ", options=age_band_driver)
        f_sex_of_driver=st.selectbox("Select the Gender of Driver: ", options=sex_of_driver)
        f_driver_education=st.selectbox("Select the Education Driver: ", options=driver_education)
        f_vehicle_driver_relation=st.selectbox("Select the Vehicle driver Relation: ", options=vehicle_driver_relation)
        f_driving_experience=st.selectbox("Select the Driving Experience: ", options=driving_experience)
        f_type_of_vehicle=st.selectbox("Select the Type of Vehicle: ", options=type_of_vehicle)
        f_owner_of_vehicle=st.selectbox("Select the Owner of Vehicle: ", options=owner_of_vehicle)
        f_service_year_of_vehicle=st.selectbox("Select the Service Year of Vehicle: ", options=service_year_of_vehicle)
        f_area_accident_occured=st.selectbox("Select the Area of accident: ", options=area_accident_occured)
        f_lanes_or_medians=st.selectbox("Select the Area of accident: ", options=lanes_or_medians)
        f_road_allignment=st.selectbox("Select the Road Alignment: ", options=road_allignment)
        f_types_of_junction=st.selectbox("Select the Type of Junction: ", options=types_of_junction)
        f_road_surface_type=st.selectbox("Select the Road Surface Type: ", options=road_surface_type)
        f_road_surface_conditions=st.selectbox("Select the Road Surface Conditions: ", options=road_surface_conditions)
        f_light_conditions=st.selectbox("Select the Light Conditions: ", options=light_conditions)
        f_weather_conditions=st.selectbox("Select the Weather Conditions: ", options=weather_conditions)
        f_type_of_collision=st.selectbox("Select the Collision Type: ", options=type_of_collision)
        f_vehicle_movement=st.selectbox("Select the Vehicle Movement: ", options=vehicle_movement)
        f_casualty_class=st.selectbox("Select the Casualty Class: ", options=casualty_class)
        f_sex_of_casualty=st.selectbox("Select the Casualty Gender: ", options=sex_of_casualty)
        f_age_band_of_casualty=st.selectbox("Select the Casualty Age Group: ", options=age_band_of_casualty)
        f_work_of_casuality=st.selectbox("Select the Work of Casuality: ", options=work_of_casuality)
        f_fitness_of_casuality=st.selectbox("Select the Fitness of Casuality",options=fitness_of_casuality)
        f_pedestrian_movement=st.selectbox("Select the Pedestrian Movement",options=pedestrian_movement)
        f_cause_of_accident=st.selectbox("Select the Accident Cause",options=cause_of_accident)
        f_number_of_casualties=st.slider("Number of Casualties",1,8,value=0,format="%d")
        f_casualties_severity=st.slider("Casualties Severity",0,3,value=0,format="%d")
        f_number_of_vehicles_invovled=st.slider("Number of vehicles involved",1,7,value=0,format="%d")        
        submit = st.form_submit_button("Predict")


    if submit:
        f_day_of_week = get_encoded_value(f_day_of_week,'Day_of_week',encodings)
        f_casualty_fitness = get_encoded_value(f_casualty_fitness,'Fitness_of_casuality',encodings)
        f_age_band_driver= get_encoded_value(f_age_band_driver,'Age_band_of_driver',encodings)
        f_sex_of_driver=get_encoded_value(f_sex_of_driver,'Sex_of_driver',encodings)
        f_driver_education=get_encoded_value(f_driver_education,'Educational_level',encodings)
        f_vehicle_driver_relation=get_encoded_value(f_vehicle_driver_relation,'Vehicle_driver_relation',encodings)
        f_driving_experience=get_encoded_value(f_driving_experience,'Driving_experience',encodings)
        f_type_of_vehicle=get_encoded_value(f_type_of_vehicle,'Type_of_vehicle',encodings)
        f_owner_of_vehicle=get_encoded_value(f_owner_of_vehicle,'Owner_of_vehicle',encodings)
        f_service_year_of_vehicle=get_encoded_value(f_service_year_of_vehicle,'Service_year_of_vehicle',encodings)
        f_area_accident_occured=get_encoded_value(f_area_accident_occured,'Area_accident_occured',encodings)
        f_lanes_or_medians=get_encoded_value(f_lanes_or_medians,'Lanes_or_Medians',encodings)
        f_road_allignment=get_encoded_value(f_road_allignment,'Road_allignment',encodings)
        f_types_of_junction=get_encoded_value(f_types_of_junction,'Types_of_Junction',encodings)
        f_road_surface_type=get_encoded_value(f_road_surface_type,'Road_surface_type',encodings)
        f_road_surface_conditions=get_encoded_value(f_road_surface_conditions,'Road_surface_conditions',encodings)
        f_light_conditions=get_encoded_value(f_light_conditions,'Light_conditions',encodings)
        f_weather_conditions=get_encoded_value(f_weather_conditions,'Weather_conditions',encodings)
        f_type_of_collision=get_encoded_value(f_type_of_collision,'Type_of_collision',encodings)
        f_vehicle_movement=get_encoded_value(f_vehicle_movement,'Vehicle_movement',encodings)
        f_casualty_class=get_encoded_value(f_casualty_class,'Casualty_class',encodings)
        f_sex_of_casualty=get_encoded_value(f_sex_of_casualty,'Sex_of_casualty',encodings)
        f_age_band_of_casualty=get_encoded_value(f_age_band_of_casualty,'Age_band_of_casualty',encodings)
        f_work_of_casuality=get_encoded_value(f_work_of_casuality,'Work_of_casuality',encodings)
        f_fitness_of_casuality=get_encoded_value(f_fitness_of_casuality,'Fitness_of_casuality',encodings)
        f_pedestrian_movement=get_encoded_value(f_pedestrian_movement,'Pedestrian_movement',encodings)
        f_cause_of_accident=get_encoded_value(f_cause_of_accident,'Cause_of_accident',encodings)



        data = np.array([f_hour,f_day_of_week,f_age_band_driver,f_sex_of_driver,f_driver_education,f_vehicle_driver_relation,
        f_driving_experience,f_type_of_vehicle,f_owner_of_vehicle,f_service_year_of_vehicle,
f_area_accident_occured,f_lanes_or_medians,f_road_allignment,f_types_of_junction,f_road_surface_type,
f_road_surface_conditions,f_light_conditions,f_weather_conditions,f_type_of_collision,
f_number_of_vehicles_invovled,f_number_of_casualties,f_vehicle_movement,f_casualty_class,f_sex_of_casualty,
f_age_band_of_casualty,f_casualties_severity,f_work_of_casuality,f_fitness_of_casuality,f_pedestrian_movement,
f_cause_of_accident]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity is:  {pred}")

if __name__ == '__main__':
    main()

                            
                                
                                               
                              
                   
         
          
    


           


