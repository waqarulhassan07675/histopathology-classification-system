import plotly.graph_objects as go
import pandas as pd


# =====================================================
# Probability Donut Chart
# =====================================================

def probability_chart(probability):

    malignant = probability * 100

    benign = 100 - malignant

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Benign", "Malignant"],
                values=[benign, malignant],
                hole=0.60,
                marker=dict(
                    colors=[
                        "#2ECC71",
                        "#E74C3C",
                    ]
                ),
            )
        ]
    )

    fig.update_layout(
        title="Prediction Probability",
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20,
        ),
    )

    return fig


# =====================================================
# Model Comparison Chart
# =====================================================

def comparison_chart():

    df = pd.DataFrame({

        "Model":[

            "DenseNet201",
            "VGG16",
            "MobileNetV2",
            "ResNet50",
            "Linear SVM",

        ],

        "Accuracy":[

            93.09,
            87.11,
            86.52,
            83.49,
            72.70,

        ]

    })

    fig = go.Figure()

    fig.add_bar(

        x=df["Model"],

        y=df["Accuracy"],

        text=df["Accuracy"],

        textposition="outside",

        marker_color=[

            "#2E86DE",
            "#3498DB",
            "#5DADE2",
            "#85C1E9",
            "#BDC3C7",

        ]

    )

    fig.update_layout(

        title="Model Accuracy Comparison",

        yaxis_title="Accuracy (%)",

        height=420,

    )

    return fig


# =====================================================
# Performance Table
# =====================================================

def performance_table():

    return pd.DataFrame({

        "Model":[

            "DenseNet201",
            "VGG16",
            "MobileNetV2",
            "ResNet50",
            "Linear SVM",

        ],

        "Accuracy (%)":[

            93.09,
            87.11,
            86.52,
            83.49,
            72.70,

        ],

        "Status":[

            "🥇 Best",
            "Very Good",
            "Very Good",
            "Good",
            "Baseline",

        ]

    })
