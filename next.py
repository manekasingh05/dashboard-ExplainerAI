import dash_html_components as html
import dash_bootstrap_components as dbc
from explainerdashboard import ClassifierExplainer
from explainerdashboard.custom import *
from explainerdashboard import ExplainerDashboard

class CustomModelTab(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Titanic Explainer")
        self.precision = PrecisionComponent(explainer,
                                hide_cutoff=True, hide_binsize=True,
                                hide_binmethod=True, hide_multiclass=True,
                                hide_selector=True,
                                cutoff=None)
        self.shap_summary = ShapSummaryComponent(explainer,
                                hide_title=True, hide_selector=True,
                                hide_depth=True, depth=8,
                                hide_cats=True, cats=True)
        self.shap_dependence = ShapDependenceComponent(explainer,
                                hide_title=True, hide_selector=True,
                                hide_cats=True, cats=True,
                                hide_index=True,
                                col='Fare', color_col="PassengerClass")
        self.connector = ShapSummaryDependenceConnector(
                self.shap_summary, self.shap_dependence)

    def layout(self):
        return dbc.Container([
            html.H1("Titanic Explainer"),
            dbc.Row([
                dbc.Col([
                    html.H3("Model Performance"),
                    html.Div("As you can see on the right, the model performs quite well."),
                    html.Div("The higher the predicted probability of survival predicted by"
                            "the model on the basis of learning from examples in the training set"
                            ", the higher is the actual percentage for a person surviving in "
                            "the test set"),
                ], width=4),
                dbc.Col([
                    html.H3("Model Precision Plot"),
                    self.precision.layout()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Feature Importances Plot"),
                    self.shap_summary.layout()
                ]),
                dbc.Col([
                    html.H3("Feature importances"),
                    html.Div("On the left you can check out for yourself which parameters were the most important."),
                    html.Div(f"{self.explainer.columns_ranked_by_shap(cats=True)[0]} was the most important"
                            f", followed by {self.explainer.columns_ranked_by_shap(cats=True)[1]}"
                            f" and {self.explainer.columns_ranked_by_shap(cats=True)[2]}."),
                    html.Div("If you select 'detailed' you can see the impact of that variable on "
                            "each individual prediction. With 'aggregate' you see the average impact size "
                            "of that variable on the finale prediction."),
                    html.Div("With the detailed view you can clearly see that the the large impact from Sex "
                            "stems both from males having a much lower chance of survival and females a much "
                            "higher chance.")
                ], width=4)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Relations between features and model output"),
                    html.Div("In the plot to the right you can see that the higher the priace"
                            "of the Fare that people paid, the higher the chance of survival. "
                            "Probably the people with more expensive tickets were in higher up cabins, "
                            "and were more likely to make it to a lifeboat."),
                    html.Div("When you color the impacts by the PassengerClass, you can clearly see that "
                            "the more expensive tickets were mostly 1st class, and the cheaper tickets "
                            "mostly 3rd class."),
                    html.Div("On the right you can check out for yourself how different features impact "
                            "the model output."),
                ], width=4),
                dbc.Col([
                    html.H3("Feature impact plot"),
                    self.shap_dependence.layout()
                ]),
            ])
        ])

ExplainerDashboard(explainer, CustomModelTab, hide_header=True).run()