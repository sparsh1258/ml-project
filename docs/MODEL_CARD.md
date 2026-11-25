# Model card

## What it is

Three LightGBM models trained together: one gives the estimate, two give the ends of its likely
range. The range is widened or narrowed by a fixed amount measured on past days, so that about 80%
of orders land inside it.

**For:** estimating total delivery time when an order is placed, in India.
**Not for:** judging or paying riders, or promising times to customers before retraining (below).

LightGBM trains it; the service runs the same trees without LightGBM, and training refuses to save
them unless the answers match to within 0.000000001 minutes. The reasons given for an estimate follow
each tree's path and credit every step to the input it asked about; they add up exactly, but other
methods would share the credit a little differently.

## Data

45,593 orders from India, 11 February to 6 April 2022. 41,953 are usable.

| Problem in the data | What we do |
|---|---|
| 3,640 orders have no location (0, 0) | Left out |
| 431 have a minus sign on a coordinate | Sign removed; the distances then look like all the others |
| 1,731 have no order time | Estimated as 10 minutes before pickup |
| 91 have an impossible age or rating | Both treated as unknown |
| 520 have vehicle condition 3, only ever on rows with bad rider details | Treated as unknown |

### It looks artificial

- Delivery times jump by about 6 minutes exactly at rider age 30, and are flat on both sides.
- Traffic is almost decided by the hour: 19:00 to 21:59 is nearly always a jam, 12:00 to 14:59 always high.
- Distances come in fixed steps, and preparation takes exactly 5, 10 or 15 minutes.

The model has learned this data well, but **its accuracy on real deliveries is unknown**. Retrain on
your own orders before relying on it. The source and licence of the data aren't recorded in this
repository; confirm them before any commercial use.

## How it was tested

The newest 10 days (28 March to 6 April, 9,916 orders) were held out. Settings and model type were
chosen on earlier days only, always predicting days that come after the ones learned from.

One caution: the choice of inputs (below) was checked on these same 10 days. Every input dropped
changed the error by 0.02 minutes or less, so the effect on the published number is small.

| | Average error | Real time in range |
|---|---|---|
| **All test orders** | **3.11 min** | **82%** |
| Traffic: low / medium / high / jam | 2.66 / 3.01 / 3.39 / 3.49 | 81–83% |
| Weather: best (cloudy) / worst (windy) | 2.71 / 3.37 | 81–84% |
| Distance: under 5 km / 10–15 km | 2.88 / 3.35 | 81–83% |
| Rider under 30 / 30 and over | 3.18 / 2.91 | 82% |
| Festival days (195 orders) | 2.85 | 82% |

Too few to judge: bicycles (12 test orders) and semi-urban areas (40).

### When details are unknown

| Unknown | Average error | Range width |
|---|---|---|
| Nothing | 3.11 min | 10.2 min |
| Rider's age and rating | 4.37 | 14.6 |
| Weather | 4.34 | 13.2 |
| Traffic | 3.45 | 11.8 |
| No rider assigned | 5.25 | 17.6 |
| Everything optional | 6.53 | 21.2 |

Ranges still contain the real time in 78–82% of orders in every one of these cases.

## Why LightGBM

Average error on the earlier days, all models on the same folds
([all 12](../reports/model_comparison.csv)):

| Model | Minutes |
|---|---|
| LightGBM | 3.08 |
| Histogram gradient boosting | 3.14 |
| XGBoost | 3.15 |
| Random forest | 3.19 |
| KNN | 3.95 |
| Linear regression | 4.80 |
| Guess the average | 7.60 |

LightGBM was tuned and the others were not, so treat small gaps as ties. Before tuning it scored
3.11, still the lowest. Training it to cope with missing inputs costs 0.01 minutes.

## Inputs

Distance, hour, rider's age and rating, vehicle and its condition, other orders carried, weather,
traffic, festival, area, order type.

Left out on purpose:
- **Pickup time and preparation time:** no effect on accuracy, and not known when an order is placed.
- **Month:** the data covers only February to April.
- **Day of week:** no effect.

### Rider's age

Age is a personal detail. Leaving it out raises the error from 3.1 to about 3.8 minutes on this
data. Decide whether your organisation should use it; it is optional in every request.

## Watching it in use

`/metrics` shows the spread of predictions, how often inputs are missing, and how many orders are
unlike the training data. Retrain when these drift, and compare estimates with real delivery times
regularly: nothing else shows whether the model is still right.
