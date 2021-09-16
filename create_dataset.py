import random
import pandas as pd
import numpy as np
from datetime import datetime

class Dataset:
    
    """Class that makes the dataset.

    At the initialization, the environment gets created. Using create_basis()
    all basis features are randomly generated from the environment. The cost
    function is added using create_calculations().
    """

    
    def __init__(self, N):
        
        """Defines all scopes that can be taken by the variables."""

        self.N = N

        # User information
        self.user = [("user_" + str(x+1), random.uniform(0, 1)) for x in range(5)]
        self.tasks_performed_today = np.arange(0, 7, 1)

        # Task information
        self.prompt_type = ["question", "task"]
        self.prompt_description_task = [
            ("close_window", 0.20), ("open_window", 0.20),
            ("enable_airco", 0.25), ("disable_airco", 0.25),
            ("turn_off_device", 0.15), ("enable_heating", 0.20),
            ("disable_heating", 0.20), ("enable_lightning", 0.15),
            ("disable_lightning", 0.15),
            ]
        self.prompt_description_question = [
            ("check_window", 0.10), ("check_weather", 0.05), ("check_lightning", 0.05),
            ("check_airco", 0.10), ("check_heating", 0.10),
            ]

        # General information
        
        self.x_coord = np.arange(0, 100, 0.5)
        self.y_coord = np.arange(0, 40, 0.5)
        self.floor = np.arange(0, 8, 1)

        # The dataset has a 2 year span
        self.date = pd.date_range("2021-01-01 00:00:00.000000", periods=2*365).tolist()
        self.team = ["team_" + str(x+1) for x in range(3)]

        # Intervention information
        self.close_to_goal = np.arange(-1, 1, 0.001)
        self.given_feedback = [True, False]

    def __get_prompt(self, prompt):
        
        """Takes the prompt_type,
        outputs a random prompt from the prompt_type.

        """
        if prompt == "task":
            return random.choice(self.prompt_description_task)
        else:
            return random.choice(self.prompt_description_question)

    def __normal_calculation(self, row):
        """Cost function that is not influenced by the date feature.

        Input is a row within the dataframe,
        output is the costs associated with it.
        """

        calc = 1  # initial weight set to 1

        calc = calc * ((row.User_weight + 1) / 2)

        dist = pow(pow(row.X_coord_task - row.X_coord_user, 2) +
                   pow(row.Y_coord_task - row.Y_coord_user, 2), 0.5)
        calc = calc * (1 - (dist / 108 * 0.25))  # 108 +/- = max diff in dist

        calc = calc * (1 - (abs(row.Floor_user - row.Floor_task) * (1/7) * 0.5))

        calc = calc * (1 - (row.Tasks_performed_today * 0.02))

        calc = calc * (1 - row.Prompt_weight)

        calc = calc * 1  # team does not influence the weight

        return calc

    def __make_date_dependent(self, weight, date):
        """Calculation of weight function, given that
        it is dependent on the date of the task being executed.

        A little dependency is created on the passage on time. A bit
        more dependency is created on the season of the year. In summertime 
        people are more eager to perform tasks and questions.

        input: current weight, date.
        output: weight dependent on date.
        """
        start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
        date_diff = (date - start_date).days

        weight = weight * (1 - (date_diff / 10000))
        
        season = abs(6 - min(date.month, 12-date.month))
        weight = weight * (1.09 - (season * 0.03))

        return weight

    def __include_interventions(self, weight, goal, feedback):
        
        """"Adds the effects from the intervention techniques 
        to the weight.

        -   Feedback works positively for the succes rate.
        -   If goalsetting is enabled, people are most motivated far from
            the goal and lose motivation once the goal approaches.

        """

        if(feedback):
            weight = weight * 1.10
        
        if(goal > 0):
            weight = weight * (1 + (0.1 * (1 - goal)))

        return weight

    def print_N(self):
        print(self.N)

    def create_basis(self):
        """Creates all basis variables before any calculation in regard to the
        weight has been performed. Returns the basis dataset.

        """
        data = pd.DataFrame()

        # User
        data["User"], data["User_weight"] = zip(*[random.choice(self.user) for i in range(self.N)])
        data["X_coord_user"] = np.random.choice(self.x_coord, size=self.N)
        data["Y_coord_user"] = np.random.choice(self.y_coord, size=self.N)
        data["Floor_user"] = np.random.choice(self.floor, size=self.N)
        data["Tasks_performed_today"] = np.random.choice(self.tasks_performed_today, size=self.N)

        # Prompt
        data["X_coord_task"] = np.random.choice(self.x_coord, size=self.N)
        data["Y_coord_task"] = np.random.choice(self.y_coord, size=self.N)
        data["Floor_task"] = np.random.choice(self.floor, size=self.N)
        data["prompt_type"] = np.random.choice(self.prompt_type, size=self.N)
        data["Prompt_desc"], data["Prompt_weight"] = zip(*[self.__get_prompt(prompt) for prompt in data["prompt_type"]])

        # General
        data["Date"] = np.random.choice(self.date, size=self.N)
        data["Team"] = np.random.choice(self.team, size=self.N)

        # Intervention
        data["Close_to_goal"] = np.random.choice(self.close_to_goal, size=self.N)
        data["Given_feedback"] = np.random.choice(self.given_feedback, size=self.N)

        return(data)

    def create_calculation(self, df, date_dependent=True):
        """Calculates the y (costs) column."""

        df["Weight"] = pd.Series(

            self.__normal_calculation(row)
            for row in df.itertuples()
        )

        if (date_dependent):
            df["Date_weight"] = [
                self.__make_date_dependent(weight, date) 
                for (weight, date) in zip(df["Weight"], df["Date"])
                ]
        df["Int_weight"] = [
            self.__include_interventions(weight, goal, feedback)
            for (weight, goal, feedback) in
            zip(df["Date_weight"], df["Close_to_goal"], df["Given_feedback"])
            ]

        return df





print("----------------------------------------------------")

environment = Dataset(100)

first_df = environment.create_basis()

environment.create_calculation(first_df)


print(first_df.head(20))

first_df.to_csv("output.csv")