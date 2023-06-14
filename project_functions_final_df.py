import geopandas as gpd
import matplotlib.pylab as plt
from shapely.geometry import Polygon
import numpy as np
import itertools
from operator import itemgetter
import pandas as pd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
import requests as rq
import json
from sklearn.neighbors import BallTree
from dbfread import DBF
from tqdm import tqdm
import random





def lines(shapefile, boundspoly):
    bound = Polygon(boundspoly)
    TF = shapefile.within(bound)
    edges_within_bound_index = TF[TF].index
    geo_tag = shapefile["geometry"][TF.values]
    
    emp_list = []
    for i in range(0,len(edges_within_bound_index)):
        segment_list = []
        for i in list(geo_tag.iloc[i].coords):
            segment_list.append(list(i))
    #if 2 < len(segment_list):
    #    continue
        emp_list.append(segment_list)
        
    #Create dataframe 
    data1 = {"ID": shapefile["edgeUID"][TF.values], "geometry": geo_tag}
    df_lines = gpd.GeoDataFrame(data1)
    return df_lines
    

def GetIdAndGeometry(id_lst, clienttoken):
    coor = []
    ID = []
    for id in tqdm (range(len(id_lst)),desc="Loading…", ascii=False, ncols=75):
        url = f"https://graph.mapillary.com/{id_lst[id]}?access_token={clienttoken}&fields=id,computed_geometry"
        response = rq.get(url)
        rqcon = response.content
        #Decode in order to get rid og bytestring 
        deco = rqcon.decode()
        result = json.loads(deco)
        #Loop to get list of coordinates and ID's 
        #Not all images has coordinates so the try/except only gets the ID's which has coordinates
        try:
            coord = result["computed_geometry"]["coordinates"]
            coor.append(coord)
            id = result["id"]
            ID.append(id)
        except KeyError:
            continue

    points = []
    for i in range(0,len(coor)):
        point = Point(coor[i])
        points.append(point)
        
    dataframepoints = {"ID": ID, "geometry": points}
    df_points = gpd.GeoDataFrame(dataframepoints)
    
    return df_points



    
def GetPointsInPoly(shapefile, clienttoken, bound, id_lst, buffer=0.0001):
    df_lines = lines(shapefile, bound)
    df_polygons = df_lines.copy()
    df_points = GetIdAndGeometry(id_lst, clienttoken)
    df_points.crs = df_lines.crs
    df_polygons.crs = df_lines.crs
    df_polygons["geometry"] = df_polygons["geometry"].buffer(buffer, cap_style = 2)
    pointInPoly = gpd.sjoin(df_points, df_polygons, how='left',predicate='within') 
    return pointInPoly


def PlotUnassignedPoints(shapefile, clienttoken, bound, id_lst, buffer = 0.0001):
    fig, ax = plt.subplots(figsize = (10,10))
    df_lines = lines(shapefile, bound)
    pointInPoly = GetPointsInPoly(shapefile, clienttoken, bound, id_lst, buffer)
    df_lines["geometry"].buffer(0.0001, cap_style = 2).plot(ax=ax,alpha=0.5)
    df_lines["geometry"].plot(ax=ax,alpha=0.5,color = "k")
    pointInPoly[pointInPoly['index_right'].isna()].plot(ax=ax,color = "r")
    
    
def StravaToDF(shapefile, bound, data):
    df_lines = lines(shapefile, bound)
    csv_df = {"ID": data["edge_uid"], "Activity": data["total_trip_count"]}
    # "Month": strava_csv["month"]
    csv_df = pd.DataFrame(data=csv_df)
    csv_df = csv_df.dropna(subset=["ID"])
    csv_df["Summed activity"] = csv_df.groupby("ID")["Activity"].transform("sum")
    collapsed_df = csv_df.drop_duplicates(subset=["ID"], keep="first")
    filtered_df = collapsed_df[collapsed_df["ID"].isin(df_lines["ID"])]
    return pd.DataFrame({"ID": filtered_df["ID"], "Activity": filtered_df["Summed activity"]})


#Helper function
def count_occurrence(lst):
    count = {}
    for item in lst:
        if item in count:
            count[item] += 1
        else:
            count[item] = 1
    return count


def GetDetections(ids, clienttoken, data_type = 'string'):# Returns df of detections per image ID and 
                                    # total unique occurences of detections of all IDs
                                    # variable dict_image (used in later functions)
    if data_type == 'string':
        list_image_id = [eval(ids[i]) for i in ids]
    else:
        list_image_id = ids
   
    ID_image = []
    detection = []
   
    for i in tqdm (range(0,len(list_image_id)),desc="Loading…", ascii=False, ncols=75):
        url = f"https://graph.mapillary.com/{list_image_id[i]}/detections?access_token={clienttoken}&fields=image,value"
        response = rq.get(url)
        rqcon = response.content
        #Decode in order to get rid of bytestring 
        deco = rqcon.decode()
        result = json.loads(deco)
        images = list(result.values())[0]
        #print(len(images))
        #print(images)
        for i in range(0,len(images)):
            try:
                val = str(images[i]["value"])
                detection.append(val)
                ID = str(images[i]["image"]["id"])
                ID_image.append(ID)
            except KeyError:
                continue
            
    #list of objects to remove 
    vd = ["void--dynamic","void--static","void--unlabeled"]

    #Removal of objects
    indices_to_remove = [i for i, string in enumerate(detection) if string in vd]
    new_string_list = []
    new_index_list = []
    for i in range(len(detection)):
        if i not in indices_to_remove:
            new_string_list.append(detection[i])
            new_index_list.append(ID_image[i])
            
    # New dataframe
    dataframevalue = {"ID": new_index_list, "Detection": new_string_list}
    df_imagess = gpd.GeoDataFrame(dataframevalue)    
    dict_image = df_imagess.groupby('ID')['Detection'].agg(list).to_dict()
    
    
    #Manipulation to count unique occurence of detections
    dict_image2 = list(dict_image.values())
    dict_image3 =sum(dict_image2, [])
    unique_occurrences = count_occurrence(dict_image3)

    #Creation of dataframe where detections for each picture ID is found
    detections_per_image = []
    dict_image_ids = []
    image_ids = list(dict_image.keys())
    for id in image_ids:
        dict_image_ids.append({'image_id': '{}'.format(id)})
        detections = count_occurrence(dict_image[id])
        detections_per_image.append(detections)
    df_detections = pd.DataFrame.from_dict(detections_per_image)
    df_image_ids = pd.DataFrame.from_dict(dict_image_ids)

    return pd.concat([df_image_ids,df_detections], axis = 1), unique_occurrences, dict_image

def GetDetections_FaF(ids, clienttoken, data_type = 'string'):# Returns df of detections per image ID and 
                                    # total unique occurences of detections of all IDs
                                    # variable dict_image (used in later functions)
    if data_type == 'string':
        list_image_id = [eval(ids[i]) for i in ids]
    else:
        list_image_id = ids
    detections_per_image = []
    dict_image_ids = []
    dict_image = {}
    #list of objects to remove 
    vd = ["void--dynamic","void--static","void--unlabeled"]
    
    for i in tqdm (range(0,len(list_image_id)),desc="Loading…", ascii=False, ncols=75):
        url = f"https://graph.mapillary.com/{list_image_id[i]}/detections?access_token={clienttoken}&fields=image,value"
        response = rq.get(url)
        rqcon = response.content
        #Decode in order to get rid of bytestring 
        deco = rqcon.decode()
        result = json.loads(deco)
        images = list(result.values())[0]
        detection = []
        for i in range(0,len(images)):
            try:
                val = str(images[i]["value"])
                detection.append(val)
                ID = str(images[i]["image"]["id"])
            except KeyError:
                continue
                
        #Remove unwanted detections
        indices_to_remove = [i for i, string in enumerate(detection) if string in vd]
        new_detections = []
        for i in range(len(detection)):
            if i not in indices_to_remove:  
                new_detections.append(detection[i])
        dict_image_ids.append({'image_id': '{}'.format(ID)})
        detections = count_occurrence(new_detections)
        detections_per_image.append(detections)
    
        dict_image[ID] = new_detections
    df_detections = pd.DataFrame.from_dict(detections_per_image)
    df_image_ids = pd.DataFrame.from_dict(dict_image_ids)
    df_grouped = pd.concat([df_image_ids, df_detections], axis = 1)
    df_final = df_grouped.groupby(['image_id']).mean().reset_index()

    return df_final, dict_image



def PlotFractionImages(list_of_unique_occurences):
    sorted_unique_occurrences = sorted(list_of_unique_occurences.items(), key=lambda x:x[1])
    x_bar, y_bar = zip(*sorted_unique_occurrences)
    y_bar = [x/sum(y_bar) for x in y_bar]
    plt.figure(figsize=(18,9))
    plt.bar(x_bar, y_bar)
    plt.xticks(rotation = 90)
    plt.title("Fraction of detections")
    plt.show()
    
def PlotAvgDetection(var_dict_image):
    encounters = []
    for keys in var_dict_image.keys():
        dab = count_occurrence(var_dict_image[keys])
        objects_found = list(dab.keys()) 
        encounters.append(objects_found)
    encounters_tolist =  sum(encounters, [])
    encounters_todict = count_occurrence(encounters_tolist)
    sorted_unique_picture_encounters = sorted(encounters_todict.items(), key=lambda x:x[1])
    x_bar, y_bar = zip(*sorted_unique_picture_encounters)
    y_bar = [x/len(var_dict_image) for x in y_bar]
    plt.figure(figsize=(18,9))
    plt.bar(x_bar, y_bar)
    plt.xticks(rotation = 90)
    plt.title("Unique occurence of a detection per image")
    plt.show()
            
def DFTransform(detections, pointInPoly, activity_df, dict_image, threshold = 0.05):
    #Assigning image ID's to edge ID's through pandas join functions
    nonna_pointInPoly = pointInPoly.dropna()
    df_image_edge = nonna_pointInPoly[["ID_left","ID_right"]]
    df_image_edge = df_image_edge.rename(columns={'ID_left':'image_id', 'ID_right':'edge_id'})
    
    merged_df = pd.merge(detections, df_image_edge, on = 'image_id')

    #Lastly joining the activity from strava for the creation of the full dataset: |image_ids|detections|edge_id|#people|
    df_activity = activity_df.rename(columns={'ID':'edge_id'})
    data =  pd.merge(merged_df, df_activity, on='edge_id')

    #Move edge_id column to location 0
    edge_id_column = data.pop('edge_id')
    data.insert(0, "edge_id", edge_id_column)
    
    #Replace NaN with 0
    data = data.fillna(0)
    
    #Get detections of interest
    encounters = []
    for keys in dict_image.keys():
        dab = count_occurrence(dict_image[keys])
        objects_found = list(dab.keys()) 
        encounters.append(objects_found)
    encounters_tolist =  sum(encounters, [])
    encounters_todict = count_occurrence(encounters_tolist)
    sorted_unique_picture_encounters = sorted(encounters_todict.items(), key=lambda x:x[1])
    df_unique_detections=pd.DataFrame(list(sorted_unique_picture_encounters),columns=['detection','occurence'])
    df_unique_detections = df_unique_detections[df_unique_detections['occurence'] >= threshold*(sum(df_unique_detections['occurence']))]
    detections_of_interest = df_unique_detections['detection'].to_list()
    
    #Now we have a list with the detections that we are interested in. We insert the coloumn names for edge ID, image ID 
    #and activity to get that from our data dataframe
    detections_of_interest.insert(0, 'edge_id')
    detections_of_interest.insert(1, 'image_id')
    detections_of_interest.append('Activity')
    
    #Thus getting the desired columns for our data
    data = data[detections_of_interest]
    
    #Get the average of detections per edge
    averaged_data = data
    averaged_data = averaged_data.loc[:, averaged_data.columns != 'image_id']
    averaged_data = averaged_data.loc[:, averaged_data.columns != 'Activity']
    averaged_data=  averaged_data.groupby('edge_id').mean().reset_index()


    #Adding activity of each of the edge
    data_without_image_ids = data.loc[:, data.columns != 'image_id']
    data_with_average_detections = averaged_data.merge(df_activity, on='edge_id', how='left')
       
    return data_with_average_detections


def PlotShapefile(LinesAndActivity, bound = None):
    fig, ax = plt.subplots(1, 1, figsize = (15,15))
    LinesAndActivity.plot(ax = ax)
    
    if bound != None:
        ax.set_xlim(bound[0][0], bound[2][0])
        ax.set_ylim(bound[0][1], bound[1][1])

def ReassignActivity(data):
    low = []
    medium = []
    high = []

    for i in range(0,len(data["Activity"])):
        if data["Activity"][i] < np.percentile(data.Activity, 33):
            low.append(data["Activity"][i])
        if np.percentile(data.Activity, 33) <= data["Activity"][i] < np.percentile(data.Activity, 66):
            medium.append(data["Activity"][i])
        if np.percentile(data.Activity, 66) <= data["Activity"][i]:
            high.append(data["Activity"][i])
    
    for i in range(0, len(data["Activity"])):
        if data["Activity"][i] in low: 
            data.at[i, "Activity"] = 0
        if data["Activity"][i] in medium: 
            data.at[i, "Activity"] = 1
        if data["Activity"][i] in high: 
            data.at[i, "Activity"] = 2
            
            
def GetClasses(unique_occurences):
    labels=[]
    classes={}
    for i in range(len(list(unique_occurences))):
        labels.append(list(unique_occurences)[i])
    for j in range(len(labels)):
        s=labels[j]
        splits=s.split('--')
        cat=splits[0]
        if cat!='void':
            if cat!='object'and cat!= 'nature' and cat!='construction':
                if cat not in classes:
                    classes[cat]=list(unique_occurences.values())[j]
                else:
                    classes[cat]+=list(unique_occurences.values())[j]
            elif cat=='object' or cat=='nature':
                if 'support' not in splits:
                    if splits[1] not in classes:
                        classes[splits[1]]=list(unique_occurences.values())[j]
                    else:
                        classes[splits[1]]+=list(unique_occurences.values())[j] 
                else: 
                    if 'utility' in splits: 
                        if 'utility pole' not in classes:
                            classes['utility pole']=list(unique_occurences.values())[j]
                        else:
                            classes['utility pole']+=list(unique_occurences.values())[j] 
                    elif 'traffic' in splits: 
                        if 'traffic sign frame'not in classes:
                            classes['traffic sign frame']=list(unique_occurences.values())[j]
                        else:
                            classes['traffic sign frame']+=list(unique_occurences.values())[j]   

                    else: 
                        if 'pole'not in classes:
                            classes['pole']=list(unique_occurences.values())[j]
                        else:
                            classes['pole']+=list(unique_occurences.values())[j] 
                        
            elif cat=='construction':

                if 'flat' not in splits:
                    
                    if splits[-1] not in classes:
                        classes[splits[-1]]=list(unique_occurences.values())[j]
                    else:
                        classes[splits[-1]]+=list(unique_occurences.values())[j]
                else: 
                    
                    if splits[-1] not in classes:
                        classes[splits[-1]]=list(unique_occurences.values())[j]
                    else:
                        classes[splits[-1]]+=list(unique_occurences.values())[j]


            

         
    
    return(classes)
        
        