import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

folder = 'MDataFiles_Stage2/'

team_seeds = pd.read_csv(folder+'MNCAATourneySeeds.csv')
conferences = pd.read_csv(folder+'MTeamConferences.csv')
sample_submission = pd.read_csv(folder+'MSampleSubmissionStage2.csv')
team_names = pd.read_csv(folder+'MTeams.csv')
tournament_slots = pd.read_csv(folder+'MNCAATourneySlots.csv')

regular_season_details = pd.read_csv(folder+'MRegularSeasonDetailedResults.csv')
tournament_compact_results = pd.read_csv(folder+'MNCAATourneyCompactResults.csv')

winning_team = pd.DataFrame()
losing_team = pd.DataFrame()

columns = ['Season', 'TeamID', 'Points', 'OppPoints', 'Loc', 'NumOT',
 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO',
 'Stl', 'Blk', 'PF', 'OppFGM', 'OppFGA', 'OppFGM3', 'OppFGA3', 'OppFTM', 'OppFTA', 'OppOR',
 'OppDR', 'OppAst', 'OppTO', 'OppStl', 'OppBlk', 'OppPF']

winning_team[columns] = regular_season_details[['Season', 'WTeamID', 'WScore', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA',
 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']]

winning_team['Wins'] = 1
winning_team['Losses'] = 0

losing_team[columns] = regular_season_details[['Season', 'LTeamID', 'LScore', 'WScore', 'WLoc', 'NumOT', 'LFGM', 'LFGA',
 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3',
 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]

# Getting the game location for losing team
def change_loc(loc):
    if loc == 'H':
        return 'A'
    elif loc == 'A':
        return 'H'
    else:
        return 'N'

losing_team['Loc'] = losing_team['Loc'].apply(change_loc)

losing_team['Wins'] = 0
losing_team['Losses'] = 1

# combining winning team and losing team dataframes into one
win_lose_teams = pd.concat([winning_team, losing_team])

# summing all the rows for each column for each team for every season
combined_teams = win_lose_teams.groupby(['Season','TeamID']).sum()

# Counting how many games each team played in a season
combined_teams['NumGames'] = combined_teams['Wins'] + combined_teams['Losses']

# final processed regular season data
regular_season_input = pd.DataFrame()

# finding the win ratio for each team
regular_season_input['WinRatio'] = combined_teams['Wins'] / combined_teams['NumGames']
# finding points per game for each team
regular_season_input['PointsPerGame'] = combined_teams['Points'] / combined_teams['NumGames']
# finding how many points scored against per game for each team
regular_season_input['PointsAllowedPerGame'] = combined_teams['OppPoints'] / combined_teams['NumGames']
# ratio of points scored vs points against
regular_season_input['PointsRatio'] = combined_teams['Points'] / combined_teams['OppPoints']
# overtime average
regular_season_input['OTPerGame'] = combined_teams['NumOT'] / combined_teams['NumGames']

# field goals made average
regular_season_input['FGPerGame'] = combined_teams['FGM'] / combined_teams['NumGames']
# field goal ratio
regular_season_input['FGRatio'] = combined_teams['FGM'] / combined_teams['FGA']
# average field goals conceded
regular_season_input['FGAllowedPerGame'] = combined_teams['OppFGM'] / combined_teams['NumGames']

# 3 point field goals made average
regular_season_input['FG3PerGame'] = combined_teams['FGM3'] / combined_teams['NumGames']
# 3 point field goal ratio
regular_season_input['FG3Ratio'] = combined_teams['FGM3'] / combined_teams['FGA3']
# average 3 point field goals conceded
regular_season_input['FG3AllowedPerGame'] = combined_teams['OppFGM3'] / combined_teams['NumGames']

# free throws made average
regular_season_input['FTPerGame'] = combined_teams['FTM'] / combined_teams['NumGames']
# free throws ratio
regular_season_input['FTRatio'] = combined_teams['FTM'] / combined_teams['FTA']
# average free throws conceded
regular_season_input['FTAllowedPerGame'] = combined_teams['OppFTM'] / combined_teams['NumGames']

# offensive rebound ratio
regular_season_input['ORRatio'] = combined_teams['OR'] / (combined_teams['OR'] + combined_teams['OppDR'])
# defensive rebound ratio
regular_season_input['DRRatio'] = combined_teams['DR'] / (combined_teams['DR'] + combined_teams['OppOR'])

# assists per game
regular_season_input['AstPerGame'] = combined_teams['Ast'] / combined_teams['NumGames']

# turnovers per game
regular_season_input['TOPerGame'] = combined_teams['TO'] / combined_teams['NumGames']
# steals per game
regular_season_input['StlPerGame'] = combined_teams['Stl'] / combined_teams['NumGames']
# blocks per game
regular_season_input['BlkPerGame'] = combined_teams['Blk'] / combined_teams['NumGames']
# personal fouls per game
regular_season_input['PFPerGame'] = combined_teams['PF'] / combined_teams['NumGames']

seed_dictionary = team_seeds.set_index(['Season', 'TeamID'])

tournament_input = pd.DataFrame()

win_ids = tournament_compact_results['WTeamID']
lose_ids = tournament_compact_results['LTeamID']
season = tournament_compact_results['Season']

game_winners = pd.DataFrame()
game_winners[['Season', 'Team1', 'Team2']] = tournament_compact_results[['Season', 'WTeamID', 'LTeamID']]
# Team 1 won the match
game_winners['Result'] = 1

game_losers = pd.DataFrame()
game_losers[['Season', 'Team1', 'Team2']] = tournament_compact_results[['Season', 'LTeamID', 'WTeamID']]
# Team 1 Lost the match
game_losers ['Result'] = 0

tournament_input = pd.concat([game_winners, game_losers])
# Tournament results to start from 2003 instead of 1985 to match regular season data
tournament_input = tournament_input[tournament_input['Season'] >= 2003].reset_index(drop=True)

team_one_seeds = []
team_two_seeds = []
one_conference_rank = []
two_conference_rank = []

# getting the seed for all Team1 and Team2 in tournament_input
def get_team_seed(team, team_list, tournament, seed_dict, region):
    for x in range(len(tournament)):
        idx = (tournament['Season'][x],tournament[team][x])
        seed = seed_dict.loc[idx].values[0]
        region.append(seed)
        if len(seed) == 4:
            seed = int(seed[1:-1])
        else:
            seed = int(seed[1:])
        team_list.append(seed)

get_team_one_seed = get_team_seed('Team1',team_one_seeds, tournament_input, seed_dictionary, one_conference_rank)

get_team_two_seed = get_team_seed('Team2',team_two_seeds, tournament_input, seed_dictionary, two_conference_rank)

#add Team1 seed to data
tournament_input['Team1Seed'] = team_one_seeds

#add Team2 seed to data
tournament_input['Team2Seed'] = team_two_seeds

# Stage 1 train dataframe
train_tournament_input = tournament_input[tournament_input['Season'] <= 2015].reset_index(drop=True)

# Stage 1 test dataframe to predict the last 5 seasons regular season matches
test_tournament_input = tournament_input[tournament_input['Season'] > 2015].reset_index(drop=True)

compare_teams_train = []
compare_teams_test = []

# comparing team stats and results
def compare_teams_matchup(train_or_test, list):
    for x in range(len(train_or_test)):
        idx = (train_or_test['Season'][x],train_or_test['Team1'][x])
        team_one_score = regular_season_input.loc[idx]
        team_one_score['Seed'] = train_or_test['Team1Seed'][x]

        idx = (train_or_test['Season'][x],train_or_test['Team2'][x])
        team_two_score = regular_season_input.loc[idx]
        team_two_score['Seed'] = train_or_test['Team2Seed'][x]

        compare_team = team_one_score - team_two_score
        compare_team['Result'] = train_or_test['Result'][x]
        list.append(compare_team)

compare_train_set = compare_teams_matchup(train_tournament_input, compare_teams_train)
compare_test_set = compare_teams_matchup(test_tournament_input, compare_teams_test)

# turn into pandas dataframe
compare_teams_test = pd.DataFrame(compare_teams_test)

# turn into pandas dataframe
compare_teams_train = pd.DataFrame(compare_teams_train)

correlation = round(compare_teams_train.corr(), 2)

# shuffling compare_teams_train dataset
shuffled_compare_teams_train = compare_teams_train.sample(frac=1, random_state=1)

# shuffling compare_teams_test dataset
shuffled_compare_teams_test = compare_teams_test.sample(frac=1, random_state=1)

# getting all the x values from dataframe except result
X_train = shuffled_compare_teams_train[shuffled_compare_teams_train.columns[:-1]].values

# getting all the y values from dataframe
y_train = shuffled_compare_teams_train['Result'].values

X_test = shuffled_compare_teams_test[shuffled_compare_teams_test.columns[:-1]].values
y_test = shuffled_compare_teams_test['Result'].values

# Random Forest Classifier Model
rf_model = RandomForestClassifier(random_state=1)

# Decision Tree Classifier Model
dt_model = DecisionTreeClassifier(random_state=1)

# Decision Tree Classifier Model
lr_model = LogisticRegression(random_state=1)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1500, num = 1000)]
# Number of features to consider at every split
rf_max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
rf_max_depth = [int(x) for x in np.linspace(start = 2, stop = 1000, num = 999)]
# Minimum number of samples required to split a node
rf_min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 1000, num = 999)]
# Minimum number of samples required at each leaf node
rf_min_samples_leaf = [int(x) for x in np.linspace(start = 1, stop = 1000, num = 1000)]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# The strategy used to choose the split at each node
splitter = ['best', 'random']
# Number of features to consider at every split
dt_max_features  = ['auto', 'sqrt', 'log2', None]
# Maximum number of levels in tree
dt_max_depth = [int(x) for x in np.linspace(start = 2, stop = 1000, num = 999)]
# Minimum number of samples required to split a node
dt_min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 1000, num = 999)]
# Minimum number of samples required at each leaf node
dt_min_samples_leaf = [int(x) for x in np.linspace(start = 1, stop = 1000, num = 1000)]

# Specify the norm of the penalty
penalty = ['l1', 'l2', 'elasticnet', 'none']
# Inverse of regularization strength
C = [int(x) for x in np.linspace(start = 0.1, stop = 1000, num = 5)]
# Algorithm to use in the optimization problem
solver =['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

# random forest hyperparameters to be tuned
rf_param = {'n_estimators': n_estimators,
            'max_features': rf_max_features,
            'max_depth': rf_max_depth,
            'min_samples_split': rf_min_samples_split,
            'min_samples_leaf': rf_min_samples_leaf,
            'bootstrap': bootstrap}

# decision trees hyperparameters to be tuned
dt_param = {'splitter': splitter,
            'max_features': dt_max_features,
            'max_depth': dt_max_depth,
            'min_samples_split': dt_min_samples_split,
            'min_samples_leaf': dt_min_samples_leaf}

# decision trees hyperparameters to be tuned
lr_param = {'penalty': penalty,
            'C': C,
            'solver': solver}

# search for best parameter for model and fit model
def search_fit_model(model,param,rnd_state):
    random_search = RandomizedSearchCV(estimator = model, param_distributions = param, cv = 50, verbose = 0, n_jobs = 50, random_state = int(rnd_state))
    random_search.fit(X_train, y_train)
    prediction = random_search.predict(X_test)
    return random_search.best_params_ ,round(random_search.score(X_test,y_test), 3), prediction

# Random Forest model training
rf_train = search_fit_model(rf_model, rf_param, 1)
# Decision trees model training
dt_train = search_fit_model(dt_model, dt_param, 1)
# logistic regression model training
lr_train = search_fit_model(lr_model, lr_param, 1)

#print the best paramaters for each model
print('Random Forest Model Best Hyperparameters: ' + str(rf_train[0]))
print('Decision Tree Model Best Hyperparameters: ' + str(dt_train[0]))
print('Logistic Regression Best Hyperparameters: ' + str(lr_train[0]))

# Print Accuracy of each model
print('Random Forest Model Accuracy: ' + str(rf_train[1]))
print('Decision Tree Model Accuracy: ' + str(dt_train[1]))
print('Logistic Regression Model Accuracy: ' + str(lr_train[1]))

fig = plt.figure()
models_name = ['Decision Tree', 'Random Forest', 'Logistic Regression']
values = [(float(dt_train[1])*100), (float(rf_train[1])*100), (float(lr_train[1])*100)]
plt.bar(models_name,values, width = 0.4)
plt.title('Comparing Models Accuracy', fontweight = 'bold')
plt.xlabel('Model Names', fontweight = 'bold')
plt.ylabel('Percentage', fontweight = 'bold')
plt.show()

# predictions for each model
rf_prediction = rf_train[2]
dt_prediction = dt_train[2]
lr_prediction = lr_train[2]

# creating confusion matrix to see what was predicted right and wrong for each model
def create_conf_matrix(model_prediction, model_name):
    conf_matrix = confusion_matrix(y_test, model_prediction)
    plt.figure(figsize=(10,7))
    sn.heatmap(conf_matrix, annot=True)
    plt.title('Confusion Matrix For ' + model_name + ' Model')
    plt.xlabel('Trained Model Prediction')
    plt.ylabel('Actual Result')
    plt.show()

# confusion matrix for each trained model
rf_conf_matrix = create_conf_matrix(rf_prediction, 'Random Forest')
dt_conf_matrix = create_conf_matrix(dt_prediction, 'Decision Tree')
lr_conf_matrix = create_conf_matrix(lr_prediction, 'Logistic Regression')

# Stage 2 Predicting 2022 NCAA Tournament

# Get regular season results for 2022 season
new_regular_season = regular_season_input.reset_index()
new_regular_season = new_regular_season[new_regular_season['Season'] == 2022].reset_index(drop=True)
new_regular_season = new_regular_season.set_index(['Season', 'TeamID'])

tournament_matches = pd.DataFrame()
tournament_season = []
tournament_team_one = []
tournament_team_two = []

match_detail_list = []

for value in sample_submission['ID']:
    match_details = value.split('_')
    list = []
    for x in match_details:
        list.append(int(x))
    match_detail_list.append(list)

for x in range(0, len(match_detail_list)):
    season = match_detail_list[x][0]
    team_one = match_detail_list[x][1]
    team_two = match_detail_list[x][2]

    tournament_season.append(season)
    tournament_team_one.append(team_one)
    tournament_team_two.append(team_two)

tournament_matches['GameID'] = sample_submission['ID']
tournament_matches['Season'] = tournament_season
tournament_matches['Team1ID'] = tournament_team_one
tournament_matches['Team2ID'] = tournament_team_two

tournament_one_seeds = []
tournament_two_seeds = []
one_region_seeds = []
two_region_seeds = []

new_seed_dictionary = seed_dictionary.reset_index()
new_seed_dictionary = new_seed_dictionary[new_seed_dictionary['Season'] == 2022].reset_index(drop=True)
new_seed_dictionary = new_seed_dictionary.set_index(['Season', 'TeamID'])

get_one_seeds = get_team_seed('Team1ID', tournament_one_seeds, tournament_matches, new_seed_dictionary, one_region_seeds)
get_two_seeds = get_team_seed('Team2ID', tournament_two_seeds, tournament_matches, new_seed_dictionary, two_region_seeds)

tournament_matches['Team1Seed'] = tournament_one_seeds
tournament_matches['Team2Seed'] = tournament_two_seeds

team_one_names = []
team_two_names = []
team_names_dictionary = team_names.set_index(['TeamID'])

def get_team_names(team, names_list):
    for x in range(len(tournament_matches)):
        idx = tournament_matches[team][x]
        name = team_names_dictionary.loc[idx].values[0]
        names_list.append(name)

get_one_names = get_team_names('Team1ID', team_one_names)
get_one_names = get_team_names('Team2ID', team_two_names)

tournament_matches['Team1Name'] = team_one_names
tournament_matches['Team2Name'] = team_two_names

stat_comparison = []

for x in range(len(tournament_matches)):
    idx = (tournament_matches['Season'][x],tournament_matches['Team1ID'][x])
    team_one_score = new_regular_season.loc[idx]
    team_one_score['Seed'] = tournament_matches['Team1Seed'][x]

    idx = (tournament_matches['Season'][x],tournament_matches['Team2ID'][x])
    team_two_score = new_regular_season.loc[idx]
    team_two_score['Seed'] = tournament_matches['Team2Seed'][x]

    compare_team = team_one_score - team_two_score
    stat_comparison.append(compare_team)

# turn into pandas dataframe
compare_team_stats = pd.DataFrame(stat_comparison)

X_test_final = compare_team_stats

logistic_regression = LogisticRegression(solver= 'newton-cg', penalty= 'l2', C = 750, random_state=1)
logistic_regression = logistic_regression.fit(X_train,y_train)
score = logistic_regression.score(X_test, y_test)
prediction = logistic_regression.predict(X_test_final)

tournament_matches['Team1ConfSeed'] = one_region_seeds
tournament_matches['Team2ConfSeed'] = two_region_seeds
tournament_matches['Result'] = prediction

pre_round = {'W12': [], 'X11': [], 'Y16': [], 'Z16': []}
tourn_seed_dict = new_seed_dictionary.reset_index()

for i in range(len(tourn_seed_dict)):
    if len(tourn_seed_dict['Seed'][i]) == 4:
        if tourn_seed_dict['Seed'][i].startswith('W'):
            pre_round['W12'].append(tourn_seed_dict['TeamID'][i])
        elif tourn_seed_dict['Seed'][i].startswith('X'):
            pre_round['X11'].append(tourn_seed_dict['TeamID'][i])
        elif tourn_seed_dict['Seed'][i].startswith('Y'):
            pre_round['Y16'].append(tourn_seed_dict['TeamID'][i])
        else:
            pre_round['Z16'].append(tourn_seed_dict['TeamID'][i])

int_seed = []

for i in range(len(tourn_seed_dict)):
        seed = tourn_seed_dict['Seed'][i]
        if len(seed) == 4:
            seed = int(seed[1:-1])
        else:
            seed = int(seed[1:])
        int_seed.append(seed)

tourn_seed_dict['IntSeed'] = int_seed
tourn_seed_dict = tourn_seed_dict.set_index(['TeamID'])

def search_match_winner(team_one, team_two):
    for i in range(len(tournament_matches)):
        if tournament_matches['Team1ID'][i] == int(team_one) and tournament_matches['Team2ID'][i] == int(team_two):
            if tournament_matches['Result'][i] == 0:
                return team_two, tournament_matches['Team2Name'][i]
            else:
                return team_one, tournament_matches['Team1Name'][i]
        elif tournament_matches['Team1ID'][i] == int(team_two) and tournament_matches['Team2ID'][i] == int(team_one):
            if tournament_matches['Result'][i] == 0:
                return team_one, tournament_matches['Team2Name'][i]
            else:
                return team_two, tournament_matches['Team1Name'][i]

pre_round_w = search_match_winner(pre_round['W12'][0], pre_round['W12'][1])
pre_round_x = search_match_winner(pre_round['X11'][0], pre_round['X11'][1])
pre_round_y = search_match_winner(pre_round['Y16'][0], pre_round['Y16'][1])
pre_round_z = search_match_winner(pre_round['Z16'][0], pre_round['Z16'][1])

pre_round['W12'][0] = pre_round_w[0]
pre_round['W12'][1] = pre_round_w[1]
pre_round['X11'][0] = pre_round_x[0]
pre_round['X11'][1] = pre_round_x[1]
pre_round['Y16'][0] = pre_round_y[0]
pre_round['Y16'][1] = pre_round_y[1]
pre_round['Z16'][0] = pre_round_z[0]
pre_round['Z16'][1] = pre_round_z[1]

print("Pre Round Regions Match Winners: " + str(pre_round))

# Get 2022 tournament slots
tournament_slots = tournament_slots[tournament_slots['Season'] == 2022].reset_index(drop=True)
tournament_slots = tournament_slots.set_index('Slot')

round_one_w = {'R1W1':[], 'R1W2':[], 'R1W3':[], 'R1W4':[], 'R1W5':[], 'R1W6':[], 'R1W7':[], 'R1W8':[]}
round_one_x = {'R1X1':[], 'R1X2':[], 'R1X3':[], 'R1X4':[], 'R1X5':[], 'R1X6':[], 'R1X7':[], 'R1X8':[]}
round_one_y = {'R1Y1':[], 'R1Y2':[], 'R1Y3':[], 'R1Y4':[], 'R1Y5':[], 'R1Y6':[], 'R1Y7':[], 'R1Y8':[]}
round_one_z = {'R1Z1':[], 'R1Z2':[], 'R1Z3':[], 'R1Z4':[], 'R1Z5':[], 'R1Z6':[], 'R1Z7':[], 'R1Z8':[]}

w_strongseed = []
w_weakseed = []
x_strongseed = []
x_weakseed = []
y_strongseed = []
y_weakseed = []
z_strongseed = []
z_weakseed = []

one_w_key = []
one_x_key = []
one_y_key = []
one_z_key = []

def get_keys(round_region, key_list):
    for key in round_region:
        key_list.append(key)

get_w_keys = get_keys(round_one_w, one_w_key)
get_x_keys = get_keys(round_one_x, one_x_key)
get_y_keys = get_keys(round_one_y, one_y_key)
get_z_keys = get_keys(round_one_z, one_z_key)

def get_slot_seeds(region_keys, strong_seed_list, weak_seed_list):
    for key in range(len(region_keys)):
        strong_seed = tournament_slots.loc[region_keys[key]].values[1]
        weak_seed = tournament_slots.loc[region_keys[key]].values[2]
        strong_seed_list.append(strong_seed)
        weak_seed_list.append(weak_seed)

get_w_seeds = get_slot_seeds(one_w_key, w_strongseed, w_weakseed)
get_x_seeds = get_slot_seeds(one_x_key, x_strongseed, x_weakseed)
get_y_seeds = get_slot_seeds(one_y_key, y_strongseed, y_weakseed)
get_z_seeds = get_slot_seeds(one_z_key, z_strongseed, z_weakseed)

def assign_key(dictionary,strongseed_list, weakseed_list):
    i = 0
    for key in dictionary:
        dictionary[key] = [strongseed_list[i],weakseed_list[i]]
        i += 1

assign_keys_w = assign_key(round_one_w, w_strongseed, w_weakseed)
assign_keys_x = assign_key(round_one_x, x_strongseed, x_weakseed)
assign_keys_y = assign_key(round_one_y, y_strongseed, y_weakseed)
assign_keys_z = assign_key(round_one_z, z_strongseed, z_weakseed)

tourn_seed_dict = new_seed_dictionary.reset_index()

def change_seed_to_id(round_dictionary):
    for value in round_dictionary.values():
        for i in range(len(tourn_seed_dict)):
            if tourn_seed_dict['Seed'][i] == value[0]:
                value[0] = tourn_seed_dict['TeamID'][i]
            elif tourn_seed_dict['Seed'][i] == value[1]:
                value[1] = tourn_seed_dict['TeamID'][i]

get_one_w_teamid = change_seed_to_id(round_one_w)
get_one_x_teamid = change_seed_to_id(round_one_x)
get_one_y_teamid = change_seed_to_id(round_one_y)
get_one_z_teamid = change_seed_to_id(round_one_z)

round_one_w['R1W5'][1] = pre_round['W12'][0]
round_one_x['R1X6'][1] = pre_round['X11'][0]
round_one_y['R1Y1'][1] = pre_round['Y16'][0]
round_one_z['R1Z1'][1] = pre_round['Z16'][0]

tourn_seed_dict['IntSeed'] = int_seed
tourn_seed_dict = tourn_seed_dict.set_index(['TeamID'])

def round_match_winner(round_dictionary):
    for value in round_dictionary.values():
        search_match = search_match_winner(value[0],value[1])
        value[0] = search_match[0]
        value[1] = search_match[1]

round_one_winners_w = round_match_winner(round_one_w)
round_one_winners_x = round_match_winner(round_one_x)
round_one_winners_y = round_match_winner(round_one_y)
round_one_winners_z = round_match_winner(round_one_z)

print("Round 1 Match Winners Region W: " + str(round_one_w))
print("Round 1 Match Winners Region X: " + str(round_one_x))
print("Round 1 Match Winners Region Y: " + str(round_one_y))
print("Round 1 Match Winners Region Z: " + str(round_one_z))


round_two_w = {'R2W1':[], 'R2W2':[], 'R2W3':[], 'R2W4':[]}
round_two_x = {'R2X1':[], 'R2X2':[], 'R2X3':[], 'R2X4':[]}
round_two_y = {'R2Y1':[], 'R2Y2':[], 'R2Y3':[], 'R2Y4':[]}
round_two_z = {'R2Z1':[], 'R2Z2':[], 'R2Z3':[], 'R2Z4':[]}

two_w_strongseed = []
two_w_weakseed = []
two_x_strongseed = []
two_x_weakseed = []
two_y_strongseed = []
two_y_weakseed = []
two_z_strongseed = []
two_z_weakseed = []

two_w_key = []
two_x_key = []
two_y_key = []
two_z_key = []

get_w_keys = get_keys(round_two_w, two_w_key)
get_x_keys = get_keys(round_two_x, two_x_key)
get_y_keys = get_keys(round_two_y, two_y_key)
get_z_keys = get_keys(round_two_z, two_z_key)

get_w_seeds = get_slot_seeds(two_w_key, two_w_strongseed, two_w_weakseed)
get_x_seeds = get_slot_seeds(two_x_key, two_x_strongseed, two_x_weakseed)
get_y_seeds = get_slot_seeds(two_y_key, two_y_strongseed, two_y_weakseed)
get_z_seeds = get_slot_seeds(two_z_key, two_z_strongseed, two_z_weakseed)

assign_keys_w = assign_key(round_two_w, two_w_strongseed, two_w_weakseed)
assign_keys_x = assign_key(round_two_x, two_x_strongseed, two_x_weakseed)
assign_keys_y = assign_key(round_two_y, two_y_strongseed, two_y_weakseed)
assign_keys_z = assign_key(round_two_z, two_z_strongseed, two_z_weakseed)

def get_teamid(current_round, previous_round):
    for value in current_round.values():
        for key in previous_round.keys():
            if value[0] == key:
                value[0] = previous_round[key][0]
            elif value[1] == key:
                value[1] = previous_round[key][0]

get_two_w_teamid = get_teamid(round_two_w, round_one_w)
get_two_x_teamid = get_teamid(round_two_x, round_one_x)
get_two_y_teamid = get_teamid(round_two_y, round_one_y)
get_two_z_teamid = get_teamid(round_two_z, round_one_z)

round_two_winners_w = round_match_winner(round_two_w)
round_two_winners_x = round_match_winner(round_two_x)
round_two_winners_y = round_match_winner(round_two_y)
round_two_winners_z = round_match_winner(round_two_z)

print("Round 2 Match Winners Region W: " + str(round_two_w))
print("Round 2 Match Winners Region X: " + str(round_two_x))
print("Round 2 Match Winners Region Y: " + str(round_two_y))
print("Round 2 Match Winners Region Z: " + str(round_two_z))

round_three_w = {'R3W1':[], 'R3W2':[]}
round_three_x = {'R3X1':[], 'R3X2':[]}
round_three_y = {'R3Y1':[], 'R3Y2':[]}
round_three_z = {'R3Z1':[], 'R3Z2':[]}

three_w_strongseed = []
three_w_weakseed = []
three_x_strongseed = []
three_x_weakseed = []
three_y_strongseed = []
three_y_weakseed = []
three_z_strongseed = []
three_z_weakseed = []

three_w_key = []
three_x_key = []
three_y_key = []
three_z_key = []

get_w_keys = get_keys(round_three_w, three_w_key)
get_x_keys = get_keys(round_three_x, three_x_key)
get_y_keys = get_keys(round_three_y, three_y_key)
get_z_keys = get_keys(round_three_z, three_z_key)

get_w_seeds = get_slot_seeds(three_w_key, three_w_strongseed, three_w_weakseed)
get_x_seeds = get_slot_seeds(three_x_key, three_x_strongseed, three_x_weakseed)
get_y_seeds = get_slot_seeds(three_y_key, three_y_strongseed, three_y_weakseed)
get_z_seeds = get_slot_seeds(three_z_key, three_z_strongseed, three_z_weakseed)

assign_keys_w = assign_key(round_three_w, three_w_strongseed, three_w_weakseed)
assign_keys_x = assign_key(round_three_x, three_x_strongseed, three_x_weakseed)
assign_keys_y = assign_key(round_three_y, three_y_strongseed, three_y_weakseed)
assign_keys_z = assign_key(round_three_z, three_z_strongseed, three_z_weakseed)

get_three_w_teamid = get_teamid(round_three_w, round_two_w)
get_three_x_teamid = get_teamid(round_three_x, round_two_x)
get_three_y_teamid = get_teamid(round_three_y, round_two_y)
get_three_z_teamid = get_teamid(round_three_z, round_two_z)

round_three_winners_w = round_match_winner(round_three_w)
round_three_winners_x = round_match_winner(round_three_x)
round_three_winners_y = round_match_winner(round_three_y)
round_three_winners_z = round_match_winner(round_three_z)

print("Round 3 Match Winners Region W: " + str(round_three_w))
print("Round 3 Match Winners Region X: " + str(round_three_x))
print("Round 3 Match Winners Region Y: " + str(round_three_y))
print("Round 3 Match Winners Region Z: " + str(round_three_z))

round_four_w = {'R4W1':[]}
round_four_x = {'R4X1':[]}
round_four_y = {'R4Y1':[]}
round_four_z = {'R4Z1':[]}

four_w_strongseed = []
four_w_weakseed = []
four_x_strongseed = []
four_x_weakseed = []
four_y_strongseed = []
four_y_weakseed = []
four_z_strongseed = []
four_z_weakseed = []

four_w_key = []
four_x_key = []
four_y_key = []
four_z_key = []

get_w_keys = get_keys(round_four_w, four_w_key)
get_x_keys = get_keys(round_four_x, four_x_key)
get_y_keys = get_keys(round_four_y, four_y_key)
get_z_keys = get_keys(round_four_z, four_z_key)

get_w_seeds = get_slot_seeds(four_w_key, four_w_strongseed, four_w_weakseed)
get_x_seeds = get_slot_seeds(four_x_key, four_x_strongseed, four_x_weakseed)
get_y_seeds = get_slot_seeds(four_y_key, four_y_strongseed, four_y_weakseed)
get_z_seeds = get_slot_seeds(four_z_key, four_z_strongseed, four_z_weakseed)

assign_keys_w = assign_key(round_four_w, four_w_strongseed, four_w_weakseed)
assign_keys_x = assign_key(round_four_x, four_x_strongseed, four_x_weakseed)
assign_keys_y = assign_key(round_four_y, four_y_strongseed, four_y_weakseed)
assign_keys_z = assign_key(round_four_z, four_z_strongseed, four_z_weakseed)

get_three_w_teamid = get_teamid(round_four_w, round_three_w)
get_three_x_teamid = get_teamid(round_four_x, round_three_x)
get_three_y_teamid = get_teamid(round_four_y, round_three_y)
get_three_z_teamid = get_teamid(round_four_z, round_three_z)

round_four_winners_w = round_match_winner(round_four_w)
round_four_winners_x = round_match_winner(round_four_x)
round_four_winners_y = round_match_winner(round_four_y)
round_four_winners_z = round_match_winner(round_four_z)

print("Region W Final Winner: " + str(round_four_w))
print("Region X Final Winner: " + str(round_four_x))
print("Region Y Final Winner: " + str(round_four_y))
print("Region Z Final Winner: " + str(round_four_z))

round_five_wx = {'R5WX': []}
round_five_yz = {'R5YZ': []}

wx_strongseed = []
wx_weakseed = []
yz_strongseed = []
yz_weakseed = []

wx_key = []
yz_key = []

get_wx_keys = get_keys(round_five_wx, wx_key)
get_yz_keys = get_keys(round_five_yz, yz_key)

get_wx_seeds = get_slot_seeds(wx_key, wx_strongseed, wx_weakseed)
get_yz_seeds = get_slot_seeds(yz_key, yz_strongseed, yz_weakseed)

assign_keys_wx = assign_key(round_five_wx, wx_strongseed, wx_weakseed)
assign_keys_yz = assign_key(round_five_yz, yz_strongseed, yz_weakseed)

get_wx_teamid = get_teamid(round_five_wx, round_four_w)
get_wx_teamid = get_teamid(round_five_wx, round_four_x)
get_yz_teamid = get_teamid(round_five_yz, round_four_y)
get_wx_teamid = get_teamid(round_five_yz, round_four_z)

winners_wx = round_match_winner(round_five_wx)
winners_yz = round_match_winner(round_five_yz)

print("Semifinal Region WX Winner: " + str(round_five_wx))
print("Semifinal Region YZ Winner: " + str(round_five_yz))

championship_round = {'R6CH': []}

champ_strongseed = []
champ_weakseed = []

champ_key = []

get_champ_keys = get_keys(championship_round, champ_key)

get_champ_seeds = get_slot_seeds(champ_key, champ_strongseed, champ_weakseed)

assign_keys_champ = assign_key(championship_round, champ_strongseed, champ_weakseed)

get_wx_teamid = get_teamid(championship_round, round_five_wx)
get_wx_teamid = get_teamid(championship_round, round_five_yz)

championship_winner = round_match_winner(championship_round)

print("Championship Winner: " + str(championship_round))
