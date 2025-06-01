import urllib
import os
from pathlib import Path
from typing import List, Union, Dict
from collections import defaultdict
import datetime
from copy import deepcopy

import pandas as pd
import numpy as np

import betting

#################
base_path_data = 'https://www.football-data.co.uk/mmz4281'
FTR2Points = {'H': 3, 'D': 1, 'A': 0}  # Number of points won by the home team of a match ('H' for 'Home', 'D' for


# 'Draw', 'A' for 'Away')
#################


def get_season_ids(
        start_season: str,
        end_season: str,
        offset: Union[List[int], None] = None) -> List[str]:
    """
    :param start_season: Two digit year indicating the first season analyzed, e.g. 04 for the 2004/2005 season.
    :param end_season: Two digit year indicating the last season analyzed, e.g. 05 for the 2004/2005 season.
    :param offset: Number of seasons to offset for both start and end seasons, e.g. if [-1, 1] is provided along the
     above arguments examples, then the seasons 2003/2004 (one season sooner) up to 2005/2006 will be considered. If
     not provided, no offset is applied.
    :return: The IDs of all the league seasons between *start_season* and *end_season*
    """
    start_season, end_season = int(start_season), int(end_season)

    # 修复：处理跨世纪的情况
    if start_season > end_season:  # Starting season is in the XXe century, end season is in XXIe century
        end_season += 100

    if offset is None:
        offset = [0, 0]

    seasons = []
    # 修复：确保至少包含起始和结束季节
    for year in range(start_season + offset[0], end_season + offset[1] + 1):  # +1 to include end_season
        season_id = '%s%s' % (str(year % 100).zfill(2), str((year + 1) % 100).zfill(2))
        seasons.append(season_id)
        print(f"Debug: Added season {season_id} (year {year})")

    print(f"Debug: get_season_ids({start_season}, {end_season}, {offset}) -> {seasons}")
    return seasons


class League(object):
    def __init__(self, betting_platforms: List[str], **kwargs):
        """
        :param betting_platforms: List of betting platforms tickers, e.g. 'BW' for Bet&Win platform.
        :param kwargs: Parsed main file arguments
        """
        self.country = kwargs['country']
        print('Country: %s' % self.country)
        self.division = kwargs['division']
        print('Division: %s' % self.division)

        # 确定国家代码
        id_country = self.country[0].upper()
        if self.country.lower() == 'spain':
            id_country = 'SP'
        elif self.country.lower() == 'germany':
            id_country = 'D'
        elif self.country.lower() == 'england':
            self.division -= 1  # to follow the id of the website from which we pull the results

        self.name = id_country + str(self.division)
        print(f'League identifier: {self.name}')

        match_historic = [] if kwargs['number_previous_direct_confrontations'] else None

        seasons = get_season_ids(kwargs['start_season'], kwargs['end_season'])

        # 修复：检查seasons是否为空
        if not seasons:
            raise ValueError(
                f"No seasons found for start_season={kwargs['start_season']} and end_season={kwargs['end_season']}")

        print("Analyzing the seasons from %s to %s..." % (seasons[0], seasons[-1]))

        # 初始化seasons列表，但先不加载数据
        self.seasons = []
        self.datasets = {}
        self.betting_platforms = betting_platforms

        # 逐个创建season，并处理可能的数据缺失
        for season_id in seasons:
            try:
                season = Season(self.name, season_id, match_historic, betting_platforms, **kwargs)
                if not season.matches.empty:  # 只添加有数据的季节
                    self.seasons.append(season)
                    print(f"✅ Successfully loaded season {season_id} with {len(season.matches)} matches")
                else:
                    print(f"⚠️ Season {season_id} has no match data, skipping...")
            except Exception as e:
                print(f"⚠️ Failed to load season {season_id}: {e}")
                continue

        if not self.seasons:
            raise ValueError(f"No valid seasons found for {self.country} division {kwargs['division']} "
                             f"between {kwargs['start_season']} and {kwargs['end_season']}")

        print(f"Successfully loaded {len(self.seasons)} seasons: {[s.name for s in self.seasons]}")

    def run(self):
        """
        :return:

        Run the matches for all seasons to gather a dataset for the ML model training and testing
        """
        for season in self.seasons:
            print(f"Running season {season.name}...")
            season.run()
            self.datasets[season.name] = season.dataset
            print(f"Season {season.name} completed with {len(season.dataset)} matches")

    def analyze_betting_platforms_margins(self):
        """
        :return:

        Analyze the average margins of betting platforms by summing the inverse of their home, away and draw odds.
        """
        margins = {}
        all_matches = pd.concat([season.matches for season in self.seasons])
        output = 'Average margin of each betting platform per match:'
        for platform in self.betting_platforms:
            odd_tickers = {platform + result for result in ['H', 'D', 'A']}
            if len(odd_tickers.intersection(all_matches.columns)) == 3:
                odds = all_matches.loc[:, list(odd_tickers)].dropna()
                if not odds.empty:
                    inv_odds = 1.0 / odds
                    probs = inv_odds.sum(axis=1)
                    margins[platform] = probs.mean()
                    output = output + ' %s: %.1f%%,' % (platform, 100 * margins[platform] - 100)
                else:
                    margins[platform] = np.nan
            else:
                margins[platform] = np.nan

        valid_margins = [m for m in margins.values() if not np.isnan(m)]
        if valid_margins:
            margins['average'] = np.mean(valid_margins)
            print(output + ' average: %.1f%%' % (100 * margins['average'] - 100))
        else:
            print("No valid margins found for any betting platform")


class Season(object):
    def __init__(
            self,
            league_name: str,
            name: str,
            match_historic: Union[List, None],
            betting_platforms: List[str],
            **kwargs):
        """
        :param league_name: Name of the league, e.g. 'SP1' for 1st Spanish division
        :param name: Four digits ID of the season, e.g. '0405' for the 2004/2005 season
        :param match_historic: List of matches from previous seasons that were already loaded. None can also be passed
        if the previous matches are not needed.
        :param betting_platforms: List of betting platforms tickers, e.g. 'BW' for Bet&Win platform
        :param kwargs: Parsed main file arguments
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.league_name = league_name
        self.name = name
        self.betting_platforms = betting_platforms

        # 尝试获取比赛数据
        try:
            self.matches = self.get_season_matches(name, league_name)
            print(f"Loaded {len(self.matches)} matches for season {name}")
        except Exception as e:
            print(f"Warning: Could not load matches for season {name}: {e}")
            self.matches = pd.DataFrame()  # 创建空DataFrame

        # 处理历史比赛数据
        if match_historic is not None and not self.matches.empty:
            if not len(match_historic):
                try:
                    previous_seasons = get_season_ids(
                        start_season=self.name[:2], end_season=self.name[2:],
                        offset=[-self.number_previous_direct_confrontations - 1, -1])
                    for season in previous_seasons:
                        try:
                            historic_matches = self.get_season_matches(season, league_name)
                            if not historic_matches.empty:
                                match_historic.append(historic_matches)
                        except Exception as e:
                            print(f"Warning: Could not load historic season {season}: {e}")
                            continue
                except Exception as e:
                    print(f"Warning: Could not load match historic: {e}")

            if match_historic and not self.matches.empty:
                try:
                    self.match_historic = pd.concat(match_historic, ignore_index=True)
                    match_historic.append(self.matches)
                except Exception as e:
                    print(f"Warning: Could not concatenate match historic: {e}")
                    self.match_historic = pd.DataFrame()
            else:
                self.match_historic = pd.DataFrame()
        else:
            self.match_historic = pd.DataFrame()

        self._matches = None
        self.teams = None
        self.ranking = None
        self.dataset = None
        self.clear_data()

    @staticmethod
    def get_season_matches(name: str, league_name: str) -> pd.DataFrame:
        """
        :param name: Four digits ID of the season, e.g. '0405' for the 2004/2005 season
        :param league_name: Name of the league, e.g. 'SP1' for 1st Spanish division
        :return: The season's matches data as available on the football-data.co.uk website.

        Load the match results for the given season and league. The matches are also locally saved for faster/offline
        loading during the next script executions.
        """
        season_id = '/'.join((name, league_name + '.csv'))
        local_path = '/'.join(('data', season_id))

        if os.path.exists(local_path):  # Load matches from local file
            print(f"Loading matches from local file: {local_path}")
            try:
                matches = pd.read_csv(local_path, sep=',')
            except Exception as e:
                print(f"Error reading local file {local_path}: {e}")
                raise
        else:  # Load matches from football-data.co.uk website
            data_url = '/'.join((base_path_data, season_id))
            print(f"Downloading matches from: {data_url}")
            try:
                matches = pd.read_csv(data_url, sep=',')
            except urllib.error.HTTPError as e:
                print(f'The following data URL seems incorrect: {data_url}')
                print(f'HTTP Error: {e}')
                raise Exception(f'Could not download data from {data_url}')
            except pd.errors.ParserError as err:  # extra empty columns are provided for some rows, just ignore them
                print(f"Parser error: {err}")
                try:
                    columns = pd.read_csv(data_url, sep=',', nrows=1).columns.tolist()
                    matches = pd.read_csv(data_url, sep=',', names=columns, skiprows=1)
                except Exception as e2:
                    print(f"Failed to parse with custom column names: {e2}")
                    raise

            # 保存到本地
            try:
                Path(os.path.split(local_path)[0]).mkdir(parents=True, exist_ok=True)
                matches.to_csv(local_path, index=False)
                print(f"Saved matches to local file: {local_path}")
            except Exception as e:
                print(f"Warning: Could not save to local file {local_path}: {e}")

        # 清理数据
        matches = matches.dropna(how='all')

        # 检查是否有必要的列
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR']
        missing_cols = [col for col in required_cols if col not in matches.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            # 如果缺少关键列，返回空DataFrame
            if 'Date' in missing_cols or 'HomeTeam' in missing_cols or 'AwayTeam' in missing_cols:
                return pd.DataFrame()

        def normalize_year(year: str) -> str:
            """
            :param year: Two or four digit long year
            :return: Four digit long year

            Normalize the year for 2017/2018 French 1st league since the file names on the football-data.co.uk website
             follow the DD/MM/YY format instead of the DD/MM/YYYY format used for other leagues
            """
            if len(year) == 2:
                current_year = int(str(datetime.datetime.now().year)[-2:])
                if int(year) <= current_year:
                    year = '20' + year  # XXIe century
                else:
                    year = '19' + year  # XXe century
            return year

        # Sort the matches by chronological order
        try:
            matches['day'] = matches['Date'].apply(lambda x: x.split('/')[0])
            matches['month'] = matches['Date'].apply(lambda x: x.split('/')[1])
            matches['year'] = matches['Date'].apply(lambda x: normalize_year(x.split('/')[2]))
            matches['Date'] = matches.apply(lambda df: '/'.join((df['day'], df['month'], df['year'])), axis=1)
            matches['Date'] = pd.to_datetime(matches['Date'], format='%d/%m/%Y')
            matches.sort_values(by=['Date'], inplace=True)
        except Exception as e:
            print(f"Warning: Could not parse dates properly: {e}")
            # 如果日期解析失败，至少保留数据

        return matches

    def clear_data(self):
        """
        :return:

        Clear the season data
        """
        if not self.matches.empty and 'HomeTeam' in self.matches.columns:
            team_names = self.matches['HomeTeam'].unique()
            self.teams = {team_name: Team(team_name, self.match_history_length) for team_name in team_names}
            self.ranking = self.get_ranking()
        else:
            self.teams = {}
            self.ranking = pd.DataFrame()
        self.dataset = []

    def update_statistics(self, played_matches: pd.DataFrame):
        """
        :param played_matches: Matches that were played at the current date
        :return:
        """
        required_stats = ['FTR', 'FTHG', 'FTAG']
        for stat in required_stats:
            if stat not in played_matches.columns:
                print(f"Warning: {stat} statistics not available")
                return

        for _, match in played_matches.iterrows():
            if pd.isna(match.get('FTR')) or match['FTR'] not in ['H', 'D', 'A']:
                print(f"Warning: Invalid match result: {match.get('FTR')}")
                continue

            for home_or_away in ['Home', 'Away']:
                team_key = f'{home_or_away}Team'
                if team_key in match and match[team_key] in self.teams:
                    self.teams[match[team_key]].update(match, home_or_away)

        self.ranking = self.get_ranking()

    def get_ranking(self) -> pd.DataFrame:
        """
        :return: The ranking of teams for the current date.
        """
        if not self.teams:
            return pd.DataFrame()

        ranking_props = ['name', 'played_matches', 'points', 'goal_difference', 'scored_goals', 'conceded_goals']
        ranking = pd.DataFrame([{key: value for key, value in vars(team).items() if key in ranking_props}
                                for team in self.teams.values()])
        if not ranking.empty:
            ranking.set_index('name', inplace=True)
            ranking.sort_values(['points', 'goal_difference'], ascending=False, inplace=True)
            for team in self.teams.values():
                team.ranking = 1 + ranking.index.get_loc(team.name)
        return ranking

    def run(self, betting_strategy: Union[betting.BettingStrategy, None] = None):
        """
        :param betting_strategy: Optional betting strategy to apply while running the season.
        :return:

        Run the whole season matchday by matchday and prepare a dataset for ML model training and testing.
        """
        if self.matches.empty:
            print(f"No matches available for season {self.name}")
            self.dataset = pd.DataFrame()
            return

        self.clear_data()
        self._matches = deepcopy(self.matches)

        if betting_strategy is not None:
            print('\nLeveraging the predictive models to bet for the %s season...' % self.name)

        match_count = 0
        while len(self._matches):
            # Group the matches by date
            current_date = self._matches['Date'].iloc[0]
            matches = self._matches.loc[self._matches['Date'] == current_date]
            dataset = []

            for _, match in matches.iterrows():
                try:
                    example = self.prepare_example(match)
                    dataset.append(example)
                except Exception as e:
                    print(f"Warning: Could not prepare example for match: {e}")
                    continue

            if dataset:
                dataset = pd.DataFrame(dataset, index=matches.index)

                if betting_strategy is not None:
                    try:
                        betting_strategy.apply(dataset, matches)
                        betting_strategy.record_bankroll(current_date)
                    except Exception as e:
                        print(f"Warning: Betting strategy failed: {e}")

                dataset = dataset.dropna()  # drop the matches with Nan in the features
                if not dataset.empty:
                    self.update_statistics(matches)
                    self.dataset.append(dataset)
                    match_count += len(dataset)

            self._matches = self._matches[self._matches['Date'] != current_date]

        if self.dataset:
            self.dataset = pd.concat(self.dataset, ignore_index=True)
        else:
            self.dataset = pd.DataFrame()

        print(f"Season {self.name} processed {match_count} matches")

    def prepare_example(self, match: pd.Series) -> Dict:
        """
        :param match: Data of a football match
        :return:

        Gather features and label about the match for later training and evaluating a ML model to predict the outcome
        of the match (win, loose, draw)
        """
        example = {'result': match.get('FTR', 'H')}  # ground truth, default to 'H' if missing

        # Gather numerical features for both home and away teams
        for home_or_away in ['Home', 'Away']:
            team_name = match.get(f'{home_or_away}Team')
            if not team_name or team_name not in self.teams:
                # 如果找不到队伍，设置默认值
                example[f'{home_or_away}PlayedMatches'] = 0
                example[f'{home_or_away}Ranking'] = 10
                example[f'{home_or_away}AvgPoints'] = 1.0
                continue

            team = self.teams[team_name]

            # Current league ranking features
            example[f'{home_or_away}PlayedMatches'] = team.played_matches
            example[f'{home_or_away}Ranking'] = team.ranking if team.ranking is not None else 10
            example[f'{home_or_away}AvgPoints'] = np.divide(team.points,
                                                            team.played_matches) if team.played_matches > 0 else 0

            # Features related to the most recent matches against other teams in the league
            if self.match_history_length is not None:
                for i in range(1, 1 + self.match_history_length):
                    key = f'{home_or_away}Prev{i}'
                    if i <= len(team.last_k_matches[home_or_away]):
                        prev_match = team.last_k_matches[home_or_away][-i]
                        if prev_match['Res'] == 'D':
                            coeffs = defaultdict(lambda: -1)
                            coeffs[home_or_away[0]] = 1
                        elif prev_match['Res'] == 'W':
                            coeffs = defaultdict(lambda: 0)
                            coeffs[home_or_away[0]] = 1
                        elif prev_match['Res'] == 'L':
                            coeffs = defaultdict(lambda: -1)
                            coeffs[home_or_away[0]] = 0
                        else:
                            example[key] = np.nan
                            continue

                        # Score comparing the betting odds and the actual results to gauge the team form
                        current_form_score = 0
                        for prev_home_or_away in ['H', 'A']:
                            odd_tickers = {platform + prev_home_or_away for platform in self.betting_platforms}
                            available_tickers = list(odd_tickers.intersection(prev_match.keys()))
                            if available_tickers:
                                odd_result = prev_match.loc[available_tickers].mean()
                                current_form_score += coeffs[prev_home_or_away] * odd_result
                        example[key] = current_form_score
                    else:
                        example[key] = np.nan

        # Features related to the direct confrontations of the home and away teams in the past seasons
        if self.number_previous_direct_confrontations and not self.match_historic.empty:
            home_team = match.get('HomeTeam')
            away_team = match.get('AwayTeam')

            if home_team and away_team:
                previous_confrontations = self.match_historic[
                    (self.match_historic['HomeTeam'] == home_team) &
                    (self.match_historic['AwayTeam'] == away_team)]
                previous_confrontations = previous_confrontations[-self.number_previous_direct_confrontations:]

                for i in range(1, 1 + self.number_previous_direct_confrontations):
                    key = f'PrevConfrFTR{i}'
                    if i <= len(previous_confrontations):
                        result = previous_confrontations.iloc[-i].get('FTR', 'H')
                        example[key] = result
                        if self.match_results_encoding == 'points':
                            example[key] = FTR2Points.get(result, 1)
                    else:
                        example[key] = np.nan

        return example


class Team(object):
    def __init__(self, name: str, match_history_length: Union[None, int]):
        """
        :param name: Name of the team, e.g. Man City
        :param match_history_length: Number of recent matches to keep track of
        """
        self.name = name
        self.match_history_length = match_history_length

        # Current season attributes
        self.played_matches = 0
        self.points = 0
        self.goal_difference = 0
        self.scored_goals = 0
        self.conceded_goals = 0
        self.ranking = None
        self.last_k_matches = {'Home': [], 'Away': []}

    def update(self, match: pd.Series, home_or_away: str):
        """
        :param match: Match involving the team
        :param home_or_away: Whether the team is the 'Home' or 'Away' team for this match
        :return:

        Update the team's season attributes with the input match
        """
        match = match.copy()
        self.played_matches += 1

        match_result = match.get('FTR', 'H')
        if match_result == home_or_away[0]:
            points = 3
            match['Res'] = 'W'  # win
        elif match_result == 'D':
            points = 1
            match['Res'] = 'D'
        else:
            points = 0
            match['Res'] = 'L'  # loose

        self.points += points
        match['Points'] = points

        # 更新进球数据
        home_goals = match.get(f'FT{home_or_away[0]}G', 0)
        away_goals = match.get(f'FT{"A" if home_or_away == "Home" else "H"}G', 0)

        try:
            home_goals = int(home_goals) if pd.notna(home_goals) else 0
            away_goals = int(away_goals) if pd.notna(away_goals) else 0
        except (ValueError, TypeError):
            home_goals = away_goals = 0

        self.scored_goals += home_goals
        self.conceded_goals += away_goals
        self.goal_difference = self.scored_goals - self.conceded_goals

        if self.match_history_length is not None:
            self.last_k_matches[home_or_away].append(match)
            self.last_k_matches[home_or_away] = self.last_k_matches[home_or_away][-self.match_history_length:]