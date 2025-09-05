# Understanding the GloBox Database

### Question 1: Can a user show up more than once in the activity table? Yes or no, and why?
~~~~sql
WITH usr_count AS (
  SELECT
    act.uid,
    COUNT(act.uid) AS usr_count
  FROM activity AS act
  GROUP BY 1
)

SELECT
	COUNT(usr_count.uid) AS num_usr_more_than_1_activity
FROM usr_count
WHERE usr_count.usr_count > 1;
~~~~

Output :

| num_usr_more_than_1_activity |
| ---------------------------- |
| 139                          |


### Question 2: What are the start and end dates of the experiment?
~~~~sql
SELECT
	MIN(grp.join_dt) AS start_date,
  MAX(grp.join_dt) AS end_date
FROM groups AS grp;
~~~~

Output:

| start_date | end_date   |
| ---------- | ---------- |
| 2023-01-25 | 2023-02-06 |


### Question 3: How many total users were in the experiment?
~~~~sql
SELECT
	COUNT(grp.uid) AS num_usr_in_experiment
FROM groups AS grp;
~~~~

Output:

| num_usr_in_experiment |
| --------------------- |
| 48943                 |


### Question 4: How many users were in the control and treatment groups?
~~~~sql
SELECT
	grp.group,
	COUNT(grp.uid) AS num_usr_in_experiment
FROM groups AS grp
GROUP BY 1;
~~~~

Output:

| group | num_usr_in_experiment |
| ----- | --------------------- |
| A     | 24343                 |
| B     | 24600                 |


### Question 5: What was the conversion rate of all users?
~~~~sql
WITH UserConversion AS (
  SELECT
  	DISTINCT usr.id,
    CASE
      WHEN act.spent > 0 THEN 1
      ELSE 0
    END AS usr_conversion
  FROM users AS usr
  LEFT JOIN activity AS act
    ON act.uid = usr.id
  GROUP BY usr.id, act.spent
)

SELECT
  (SUM(uc.usr_conversion)::numeric/COUNT(uc.usr_conversion)::numeric)*100 AS usr_conversion_rate
FROM UserConversion AS uc;
~~~~

Output:

| usr_conversion_rate    |
| ---------------------- |
| 4.27844635596510226200 |


### Question 6: What is the user conversion rate for the control and treatment groups?
~~~~sql
WITH UserConversion AS (
  SELECT
  	DISTINCT usr.id,
  	grp.group,
    CASE
      WHEN act.spent > 0 THEN 1
      ELSE 0
    END AS usr_conversion
  FROM users AS usr
  LEFT JOIN activity AS act
    ON act.uid = usr.id
  LEFT JOIN groups AS grp
  	ON grp.uid = usr.id
  --WHERE grp.group IS NULL
  GROUP BY 1,2, usr.id, act.spent
)

SELECT
	CASE
  	WHEN uc.group = 'A' THEN 'Control Group'
    ELSE 'Treatment Group'
  END AS usr_group,
  (SUM(uc.usr_conversion)::numeric/COUNT(uc.usr_conversion)::numeric)*100 AS usr_conversion_rate
FROM UserConversion AS uc
GROUP BY usr_group;
~~~~

Output :

| usr_group       | usr_conversion_rate    |
| --------------- | ---------------------- |
| Control Group   | 3.92309904284599268800 |
| Treatment Group | 4.63008130081300813000 |


### Question 7: What is the average amount spent per user for the control and treatment groups, including users who did not convert?
~~~~sql
WITH AvgSpent AS (
  SELECT
    usr.id,
    --act.uid,
    --grp.uid,
    grp.group,
    SUM(act.spent)::NUMERIC AS usr_spent
  FROM users AS usr
  LEFT JOIN activity AS act
    ON act.uid = usr.id
  LEFT JOIN groups AS grp
    ON grp.uid = usr.id
  --WHERE act.uid IS NOT NULL
  GROUP BY 1,2--,3,4
  ORDER BY 1
)

SELECT
	--avs.group,
  CASE
  	WHEN avs.group = 'A' THEN 'Control Group'
    WHEN avs.group = 'B' THEN 'Treatment Group'
    ELSE 'Unknown'
  END AS usr_group,
	SUM(avs.usr_spent)/COUNT(avs.id) AS avg_spent_per_user
FROM AvgSpent AS avs
GROUP BY 1;
~~~~

Output :

| usr_group       | avg_spent_per_user  |
| --------------- | ------------------- |
| Control Group   | 3.3745184679288412  |
| Treatment Group | 3.39086694588578326 |


### Question 8: Why does it matter to include users who did not convert when calculating the average amount spent per user?
~~~~sql
WITH AvgSpent AS (
  SELECT
    DISTINCT usr.id,
    --act.uid,
    --grp.uid,
    grp.group,
    SUM(act.spent)::NUMERIC AS usr_spent
  FROM users AS usr
  LEFT JOIN activity AS act
    ON act.uid = usr.id
  LEFT JOIN groups AS grp
    ON grp.uid = usr.id
  --WHERE act.uid IS NOT NULL
  GROUP BY 1,2--,3,4
  ORDER BY 1
),
UserConversion AS (
  SELECT
  	DISTINCT usr.id,
  	grp.group,
    CASE
      WHEN act.spent > 0 THEN 1
      ELSE 0
    END AS usr_conversion
  FROM users AS usr
  LEFT JOIN activity AS act
    ON act.uid = usr.id
  LEFT JOIN groups AS grp
  	ON grp.uid = usr.id
  --WHERE grp.group IS NULL
  GROUP BY 1,2, usr.id, act.spent
)
SELECT
	--avs.group,
  CASE
  	WHEN avs.group = 'A' THEN 'Control Group'
    WHEN avs.group = 'B' THEN 'Treatment Group'
    ELSE 'Unknown'
  END AS usr_group,
	SUM(avs.usr_spent)/COUNT(avs.id) AS avg_spent_per_user,
  (SUM(uc.usr_conversion)::numeric/COUNT(uc.usr_conversion)::numeric)*100 AS usr_conversion_rate
FROM AvgSpent AS avs
LEFT JOIN UserConversion AS uc
	ON uc.id = avs.id
GROUP BY 1;
~~~~

Output :

| usr_group       | avg_spent_per_user  | usr_conversion_rate    |
| --------------- | ------------------- | ---------------------- |
| Control Group   | 3.3745184679288412  | 3.92309904284599268800 |
| Treatment Group | 3.39086694588578326 | 4.63008130081300813000 |


### Question 9: Why does it matter to include users who did not convert when calculating the average amount spent per user?
~~~~sql
WITH AvgSpent AS (
  SELECT
    DISTINCT usr.id,
    --act.uid,
    --grp.uid,
    grp.group,
    SUM(act.spent)::NUMERIC AS usr_spent
  FROM users AS usr
  LEFT JOIN activity AS act
    ON act.uid = usr.id
  LEFT JOIN groups AS grp
    ON grp.uid = usr.id
  --WHERE act.uid IS NOT NULL
  GROUP BY 1,2--,3,4
  ORDER BY 1
),
UserConversion AS (
  SELECT
  	DISTINCT usr.id,
  	grp.group,
    CASE
      WHEN act.spent > 0 THEN 1
      ELSE 0
    END AS usr_conversion
  FROM users AS usr
  LEFT JOIN activity AS act
    ON act.uid = usr.id
  LEFT JOIN groups AS grp
  	ON grp.uid = usr.id
  --WHERE grp.group IS NULL
  GROUP BY 1,2, usr.id, act.spent
)
SELECT
	--avs.group,
  CASE
  	WHEN avs.group = 'A' THEN 'Control Group'
    WHEN avs.group = 'B' THEN 'Treatment Group'
    ELSE 'Unknown'
  END AS usr_group,
  COUNT(avs.id) AS num_users,
  SUM(uc.usr_conversion)::numeric AS num_users_converted,
  (SUM(uc.usr_conversion)::numeric/COUNT(uc.usr_conversion)::numeric)*100 AS usr_conversion_rate,
	SUM(avs.usr_spent)/COUNT(avs.id) AS avg_spent_per_user
FROM AvgSpent AS avs
LEFT JOIN UserConversion AS uc
	ON uc.id = avs.id
GROUP BY avs.group;
~~~~

Output :

| usr_group       | num_users | num_users_converted | usr_conversion_rate    | avg_spent_per_user  |
| --------------- | --------- | ------------------- | ---------------------- | ------------------- |
| Treatment Group | 24600     | 1139                | 4.63008130081300813000 | 3.39086694588578326 |
| Control Group   | 24343     | 955                 | 3.92309904284599268800 | 3.3745184679288412  |

--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------


#### Extracting the Analysis Dataset

### Question : Write a SQL query that returns: the user ID, the user’s country, the user’s gender, the user’s device type, the user’s test group, whether or not they converted (spent > $0), and how much they spent in total ($0+).
~~~~sql
WITH UserConv AS (
  SELECT
  	DISTINCT usr.id,
    COALESCE(usr.country, 'Unknown') AS country,
    COALESCE(usr.gender, 'Unknown') AS gender,
    COALESCE(grp.device, 'Unknown') AS device,
    COALESCE(grp.group, 'Unknown') AS "group",
    CASE
      WHEN act.spent > 0 THEN 1
      ELSE 0
    END AS user_conv
  FROM users AS usr
  LEFT JOIN groups AS grp
		ON grp.uid = usr.id
  LEFT JOIN activity AS act
		ON act.uid = usr.id
  )
SELECT
	uc.id,
  uc.country,
	uc.gender,
	uc.device,
	uc.group,
	uc.user_conv AS usr_conversion,
	SUM(CAST(COALESCE(act.spent, 0) AS DECIMAL(100, 3))) AS spent
FROM UserConv AS uc
LEFT JOIN activity AS act
	ON act.uid = uc.id
GROUP BY 1, 2, 3, 4, 5, 6
ORDER BY 1;
~~~~

Output :
| id      | country | gender  | device | group | usr_conversion | spent |
| ------- | ------- | ------- | ------ | ----- | -------------- | ----- |
| 1000000 | CAN     | M       | I      | B     | 0              | 0.000 |
| 1000001 | BRA     | M       | A      | A     | 0              | 0.000 |
| 1000002 | FRA     | M       | A      | A     | 0              | 0.000 |
| 1000003 | BRA     | M       | I      | B     | 0              | 0.000 |
| 1000004 | DEU     | F       | A      | A     | 0              | 0.000 |
| 1000005 | GBR     | F       | A      | B     | 0              | 0.000 |
| 1000006 | ESP     | M       | A      | B     | 0              | 0.000 |
| 1000007 | BRA     | F       | A      | A     | 0              | 0.000 |
| 1000008 | BRA     | F       | A      | A     | 0              | 0.000 |
| 1000009 | USA     | Unknown | A      | A     | 0              | 0.000 |
...


# Novelty Analysis in A/B Test
### First organize the data for time-series analysis with respect to conversion rate, average spent, converted users or users who paid, and total number of users. Here, is the SQL querry for that

~~~~sql
WITH DataSet AS (
  SELECT
    DISTINCT usr.id,
    SUM(CAST(COALESCE(act.spent, 0) AS DECIMAL(100, 3))) AS spent,
    CASE
      WHEN act.spent > 0 THEN 1
      ELSE 0
    END AS usr_conversion,
    COALESCE(grp.group, 'Unknown') AS group,
    --act.dt
  	grp.join_dt
  FROM users AS usr
  LEFT JOIN activity AS act
    ON act.uid = usr.id
  LEFT JOIN groups AS grp
    ON grp.uid = usr.id
  GROUP BY 1,3,4,5
  ORDER BY 1
)
SELECT
	ds.join_dt,
  ds.group,
  CAST((SUM(ds.usr_conversion)::numeric/COUNT(ds.usr_conversion)::numeric)*100 AS DECIMAL(100,3)) AS conversion_rate,
  CAST(AVG(ds.spent) AS DECIMAL(100,3)) AS avg_spent,
  SUM(ds.usr_conversion) AS num_usr_converted,
  COUNT(ds.id) AS tot_num_usr
FROM DataSet AS ds
GROUP BY ds.join_dt, ds.group
ORDER BY ds.join_dt;
~~~~

Output :
| join_dt    | group | conversion_rate | avg_spent | num_usr_converted | tot_num_usr |
| ---------- | ----- | --------------- | --------- | ----------------- | ----------- |
| 2023-01-25 | B     | 4.532           | 3.271     | 268               | 5913        |
| 2023-01-25 | A     | 3.942           | 3.293     | 226               | 5733        |
| 2023-01-26 | A     | 3.841           | 3.677     | 158               | 4114        |
| 2023-01-26 | B     | 4.909           | 3.236     | 204               | 4156        |
| 2023-01-27 | A     | 4.082           | 3.366     | 125               | 3062        |
| 2023-01-27 | B     | 4.764           | 3.830     | 142               | 2981        |
| 2023-01-28 | B     | 4.568           | 3.443     | 102               | 2233        |
| 2023-01-28 | A     | 4.762           | 3.937     | 110               | 2310        |
| 2023-01-29 | B     | 4.329           | 3.013     | 78                | 1802        |
| 2023-01-29 | A     | 3.683           | 2.815     | 65                | 1765        |
| 2023-01-30 | A     | 3.809           | 3.122     | 55                | 1444        |
| 2023-01-30 | B     | 4.483           | 3.015     | 65                | 1450        |
| 2023-01-31 | A     | 3.538           | 3.042     | 42                | 1187        |
| 2023-01-31 | B     | 4.481           | 4.398     | 54                | 1205        |
| 2023-02-01 | B     | 3.911           | 2.396     | 42                | 1074        |
| 2023-02-01 | A     | 4.273           | 3.919     | 42                | 983         |
| 2023-02-02 | A     | 3.883           | 3.325     | 36                | 927         |
| 2023-02-02 | B     | 3.995           | 2.218     | 35                | 876         |
| 2023-02-03 | A     | 3.747           | 3.194     | 29                | 774         |
| 2023-02-03 | B     | 4.566           | 4.695     | 40                | 876         |
| 2023-02-04 | A     | 2.826           | 2.549     | 21                | 743         |
| 2023-02-04 | B     | 5.517           | 3.551     | 40                | 725         |
| 2023-02-05 | B     | 4.606           | 4.060     | 31                | 673         |
| 2023-02-05 | A     | 3.167           | 3.540     | 21                | 663         |
| 2023-02-06 | A     | 3.918           | 3.139     | 25                | 638         |
| 2023-02-06 | B     | 5.975           | 3.900     | 38                | 636         |


### Second, I have calculated that how many days customer takes to be converted.
~~~~sql
WITH DataSet AS (
  SELECT
    DISTINCT usr.id,
  	grp.group,
  	CAST(EXTRACT(DAY FROM (CAST(act.dt AS TIMESTAMP) - CAST(grp.join_dt AS TIMESTAMP))) AS INTEGER) AS num_days_converted
  FROM users AS usr
  LEFT JOIN activity AS act
    ON act.uid = usr.id
  LEFT JOIN groups AS grp
    ON grp.uid = usr.id
  WHERE act.dt IS NOT NULL
  GROUP BY 1,2,3
  ORDER BY 1
)
SELECT
  ds.num_days_converted,
  CASE
  	WHEN ds.group = 'A' THEN 'Control Group'
    WHEN ds.group = 'B' THEN 'Treatment Group'
    ELSE 'Unknown'
  END AS group,
  COUNT(ds.id) AS tot_num_usr
FROM DataSet AS ds
GROUP BY num_days_converted, ds.group
ORDER BY num_days_converted;
~~~~

Output :

| num_days_converted | group           | tot_num_usr |
| ------------------ | --------------- | ----------- |
| 0                  | Control Group   | 746         |
| 0                  | Treatment Group | 902         |
| 1                  | Control Group   | 66          |
| 1                  | Treatment Group | 94          |
| 2                  | Control Group   | 55          |
| 2                  | Treatment Group | 62          |
| 3                  | Control Group   | 56          |
| 3                  | Treatment Group | 58          |
| 4                  | Control Group   | 47          |
| 4                  | Treatment Group | 47          |
| 5                  | Control Group   | 34          |
| 5                  | Treatment Group | 39          |
| 6                  | Control Group   | 10          |
| 6                  | Treatment Group | 17          |
