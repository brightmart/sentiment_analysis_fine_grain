import pandas as pd
from sklearn.metrics import f1_score, classification_report


if __name__ == "__main__":
    validation_pred = pd.read_csv("validation_rcnn_char.csv")
    validation_real = pd.read_csv("preprocess/validation_char.csv")
    f_scores = 0

    print(classification_report(validation_real["location_traffic_convenience"], validation_pred["location_traffic_convenience"]))
    print(classification_report(validation_real["location_distance_from_business_district"], validation_pred["location_distance_from_business_district"]))
    print(classification_report(validation_real["location_easy_to_find"], validation_pred["location_easy_to_find"]))
    print(classification_report(validation_real["service_wait_time"], validation_pred["service_wait_time"]))
    print(classification_report(validation_real["service_waiters_attitude"], validation_pred["service_waiters_attitude"]))
    print(classification_report(validation_real["service_parking_convenience"], validation_pred["service_parking_convenience"]))
    print(classification_report(validation_real["service_serving_speed"], validation_pred["service_serving_speed"]))
    print(classification_report(validation_real["price_level"], validation_pred["price_level"]))
    print(classification_report(validation_real["price_cost_effective"], validation_pred["price_cost_effective"]))
    print(classification_report(validation_real["price_discount"], validation_pred["price_discount"]))
    print(classification_report(validation_real["environment_decoration"], validation_pred["environment_decoration"]))
    print(classification_report(validation_real["environment_noise"], validation_pred["environment_noise"]))
    print(classification_report(validation_real["environment_space"], validation_pred["environment_space"]))
    print(classification_report(validation_real["environment_cleaness"], validation_pred["environment_cleaness"]))
    print(classification_report(validation_real["dish_portion"], validation_pred["dish_portion"]))
    print(classification_report(validation_real["dish_taste"], validation_pred["dish_taste"]))
    print(classification_report(validation_real["dish_look"], validation_pred["dish_look"]))
    print(classification_report(validation_real["dish_recommendation"], validation_pred["dish_recommendation"]))
    print(classification_report(validation_real["others_overall_experience"], validation_pred["others_overall_experience"]))
    print(classification_report(validation_real["others_willing_to_consume_again"], validation_pred["others_willing_to_consume_again"]))

    f_scores += f1_score(validation_real["location_traffic_convenience"], validation_pred["location_traffic_convenience"],
             average="macro")
    print(f1_score(validation_real["location_traffic_convenience"], validation_pred["location_traffic_convenience"],
             average="macro"))

    f_scores += f1_score(validation_real["location_distance_from_business_district"],
             validation_pred["location_distance_from_business_district"], average="macro")
    print(f1_score(validation_real["location_distance_from_business_district"],
                   validation_pred["location_distance_from_business_district"], average="macro"))

    f_scores += f1_score(validation_real["location_easy_to_find"], validation_pred["location_easy_to_find"], average="macro")
    print(f1_score(validation_real["location_easy_to_find"], validation_pred["location_easy_to_find"],
                   average="macro"))

    f_scores += f1_score(validation_real["service_wait_time"], validation_pred["service_wait_time"], average="macro")
    print(f1_score(validation_real["service_wait_time"], validation_pred["service_wait_time"],
                   average="macro"))

    f_scores += f1_score(validation_real["service_waiters_attitude"], validation_pred["service_waiters_attitude"], average="macro")
    print(f1_score(validation_real["service_waiters_attitude"], validation_pred["service_waiters_attitude"],
                   average="macro"))

    f_scores += f1_score(validation_real["service_parking_convenience"], validation_pred["service_parking_convenience"],
             average="macro")
    print(f1_score(validation_real["service_parking_convenience"], validation_pred["service_parking_convenience"],
                   average="macro"))

    f_scores += f1_score(validation_real["service_serving_speed"], validation_pred["service_serving_speed"], average="macro")
    print(f1_score(validation_real["service_serving_speed"], validation_pred["service_serving_speed"],
                   average="macro"))

    f_scores += f1_score(validation_real["price_level"], validation_pred["price_level"], average="macro")
    print(f1_score(validation_real["price_level"], validation_pred["price_level"],
                   average="macro"))

    f_scores += f1_score(validation_real["price_cost_effective"], validation_pred["price_cost_effective"], average="macro")
    print(f1_score(validation_real["price_cost_effective"], validation_pred["price_cost_effective"],
                   average="macro"))

    f_scores += f1_score(validation_real["price_discount"], validation_pred["price_discount"], average="macro")
    print(f1_score(validation_real["price_discount"], validation_pred["price_discount"],
                   average="macro"))

    f_scores += f1_score(validation_real["environment_decoration"], validation_pred["environment_decoration"], average="macro")
    print(f1_score(validation_real["environment_decoration"], validation_pred["environment_decoration"],
                   average="macro"))

    f_scores += f1_score(validation_real["environment_noise"], validation_pred["environment_noise"], average="macro")
    print(f1_score(validation_real["environment_noise"], validation_pred["environment_noise"],
                   average="macro"))

    f_scores += f1_score(validation_real["environment_space"], validation_pred["environment_space"], average="macro")
    print(f1_score(validation_real["environment_space"], validation_pred["environment_space"],
                   average="macro"))

    f_scores += f1_score(validation_real["environment_cleaness"], validation_pred["environment_cleaness"], average="macro")
    print(f1_score(validation_real["environment_cleaness"], validation_pred["environment_cleaness"],
                   average="macro"))

    f_scores += f1_score(validation_real["dish_portion"], validation_pred["dish_portion"], average="macro")
    print(f1_score(validation_real["dish_portion"], validation_pred["dish_portion"],
                   average="macro"))

    f_scores += f1_score(validation_real["dish_taste"], validation_pred["dish_taste"], average="macro")
    print(f1_score(validation_real["dish_taste"], validation_pred["dish_taste"],
                   average="macro"))

    f_scores += f1_score(validation_real["dish_look"], validation_pred["dish_look"], average="macro")
    print(f1_score(validation_real["dish_look"], validation_pred["dish_look"],
                   average="macro"))

    f_scores += f1_score(validation_real["dish_recommendation"], validation_pred["dish_recommendation"], average="macro")
    print(f1_score(validation_real["dish_recommendation"], validation_pred["dish_recommendation"],
                   average="macro"))

    f_scores += f1_score(validation_real["others_overall_experience"], validation_pred["others_overall_experience"],
             average="macro")
    print(f1_score(validation_real["others_overall_experience"], validation_pred["others_overall_experience"],
                   average="macro"))

    f_scores += f1_score(validation_real["others_willing_to_consume_again"], validation_pred["others_willing_to_consume_again"],
             average="macro")
    print(f1_score(validation_real["others_willing_to_consume_again"], validation_pred["others_willing_to_consume_again"],
                   average="macro"))

    print(f_scores / 20)