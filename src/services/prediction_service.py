#hangi prediction’ların çağrılacağını karar verir, yani modelleri çalıştırır(service gibi)
from src.predictors.umbrella_predictor import predict_umbrella
from src.predictors.suitability_predictor import predict_suitability
from src.predictors.clothing_predictor import predict_clothing
from src.predictors.activity_type_predictor import predict_activity_recommendations
from src.predictors.all_with_confidence_predictor import predict_confidence



def predict_umbrella_service(features: dict) -> dict:
    umbrella_result = predict_umbrella(features)
    return umbrella_result


def predict_suitability_service(features: dict) -> dict:
    suitablity_result=predict_suitability(features)
    return suitablity_result


def predict_clothing_service(features):
    clothing_result=predict_clothing(features)
    return clothing_result

def predict_all_service(features: dict) -> dict:
    umbrella_result = predict_umbrella(features)
    suitability_result = predict_suitability(features)
    clothing_result = predict_clothing(features)

    confidence_input = {
        **features,
        "forecasted_umbrella": umbrella_result["umbrella_needed"],
        "forecasted_clothing": clothing_result["clothing_recommendation"],
    }

    confidence_result = predict_confidence(confidence_input)

    return {
        **umbrella_result,
        **suitability_result,
        **clothing_result,
        **confidence_result,
    }


def predict_activity_service(features: dict) -> dict:
    return predict_activity_recommendations(features)


