import warnings
import Markowitz as Op
from Markowitz import *

warnings.filterwarnings("ignore")

if True:
    input_dir = "../lecture_data"
    file = "dataset.csv"
    df = pd.read_csv(os.path.join(input_dir, file))
    benchmark = pd.concat([df['Date'], df['KOSPI'], df['KOR10Y']], axis=1)
    df.drop(columns=['KOSPI', 'KOR10Y'], inplace=True)

optimal = Op.Optimization(df, benchmark)
optimal.guess_best_cov()
optimal.guess_best_mean()
optimal.optimalize_portfolio(optimal.cov, optimal.mean)
optimal.calculate_return(Plot=True)
