# United States Retail records
## Introduction
When determining which products to offer, several critical factors should be taken into account to ensure successful market entry and sustained profitability. In this project, I wil be addressing some of these factors so as to be able to choose the best product category one can offer in United States. The dataset analysed in this project is gotten from [kaggle](https://www.kaggle.com/) a popular open source for datasets. Get dataset [Here](https://www.kaggle.com/datasets/abdurraziq01/retail-data). This dataset is of retail data collected from various retailers across the United States, covering cities like Phoenix, Los Angeles, San Diego, San Jose and Houston from july 14, 2020 to july 14, 2023.

## Objectives

- Customer Demographics and Preferences:

What age group has the highest spending score?

Is there a significant difference in spending scores between genders?

What is the average annual income of customers buying different product categories?

- Product Popularity and Profitability:

Which product category has the highest sales volume?

Which product category generates the highest profit margins?

What is the average product price and how does it correlate with profit?

- Store Performance and Location Analysis:

Which store location has the highest foot traffic?

Is there a correlation between foot traffic and profit?

Which store location has the highest average profit per sale?

- Discounts and Customer Behavior:

How does applying a discount affect the profit margin?

What is the average discount percent applied across different product categories?

Do customers tend to buy more when discounts are applied?

- Competitive Analysis:

How do competitor prices affect sales volume and profit?

Which product categories are most sensitive to competitor pricing?


### Enviroment Setup
As usual, I will start with necessary imports.
```sql
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from scipy.stats import ttest_ind
```
Futher on, I will import the dataset.
```sql
df = pd.read_csv(r'C:\Users\USER\Downloads\retail_data.csv')
```
### Data Preprocessing
Now we are all set to have a first look at our data.
```sql
df.head()
```
![First look of our data](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/018e81fe-806b-4edb-9d10-e36a5bcc9434)

My first observation is the PurchaseDate column format, it contain datetime information which is in this format 2022-06-11 but I want it in this format June 11, 2022. Let's deal with that first.
```sql
# This Convert the 'PurchaseDate' column to datetime
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])

# This Format the dates to 'Month DD, YYYY'
df['FormattedDate'] = df['PurchaseDate'].dt.strftime('%B %d, %Y')

# This Drop the original 'PurchaseDate' column
df.drop(columns=['PurchaseDate'], inplace=True)

# And this Rename 'FormattedDate' column to 'PurchaseDate'
df.rename(columns={'FormattedDate': 'PurchaseDate'}, inplace=True)
```
Let take a look at the changes.
```sql
df.head()
```

![Take a look at the changes](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/834b3b4b-ed4e-4fb0-b79e-9f1e1e7d7dee)


#### Let check the Datatypes and the number of rows to ensure uniformity.
```sql
df.info()
```
![Datatypes check](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/ea414c4f-fb72-4882-8218-555f6656ddfa)

We have uniform number of rows for each column, and they have the preffered datatypes.

#### Let check for null values
```sql
df.isnull().sum()
```
![Null value check](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/1e2c9990-a884-4d3a-863b-15aa01a1bd36)

We have zero null value in each column

#### Let generate a summary of statistical metrics for the numerical columns in the DataFrame
```sql
df.describe()
```
![Statistical metrics](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/63a5078d-2075-4cc8-a006-807009fcbb74)

Before I start answering the questions from the objectives of this project, let's first check the **Age Distribution** to know the Age range with the highest frequency. 
```sql
# This will create bins with step size 2
bins = np.arange(df['Age'].min(), df['Age'].max() +3, 2)

# This function will format the counts to two decimal places
def format_counts(count):
    if count >= 1000:
        return f'{count / 1000:.2f}k'
    else:
        return str(count)

# This create an interactive histogram
fig = px.histogram(df, x='Age', nbins=len(bins), title='Age Distribution',
                   labels={'Age': 'Age'}, color_discrete_sequence=['blue'])

# This update layout for better appearance
fig.update_layout(
    title={
        'text': 'Age Distribution',
        'x': 0.5,
        'xanchor': 'center'
          },
    xaxis_title='Age',
    yaxis_title='Frequency',
    font=dict(family = 'Times New Roman', size = 20, color = 'Black'),
    bargap=0.1,
    xaxis=dict(
        tickmode='array',
        tickvals=bins)
)

# This get the counts for each bin
counts, bin_edges = np.histogram(df['Age'], bins=bins)
bin_labels = [format_counts(count) for count in counts]  # Format counts with custom function

# This add text annotations to each bar
fig.update_traces(text=bin_labels, textposition='inside', textfont_size=20)

fig.show()
```
![Age Distribution](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/55902d56-ae45-47ea-9959-7581ed5b9d1a)

The Age distribution appears to be fairly uniform between the ages of 18 and 66, with each age group having a frequency around 7.5k. This suggests that the dataset might have been sampled or collected in away that ensures an even representation of these age group. The frequency at age 70 is significantly lower(3.7k) compared to other age groups, This could indicate that the dataset has fewer samples for this age; or there might be an upper age limit in the sampling process.

### Objective 1
#### Question 1
What age group has the highest spending score?
```sql
bins = np.arange(df['Age'].min(), df['Age'].max() +2, 2)
fig = px.density_heatmap(df, x='Age', y='SpendingScore', 
                         title='Spending Score by Age', color_continuous_scale='greens', text_auto=True)

fig.update_layout(
    title={
        'text': 'Spending Score by Age',
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_title='Age',
    yaxis_title='Spending Score',
    font=dict(
        family='Times New Roman',
        size=20,
        color="black"
    ),
        coloraxis_colorbar=dict(
        title='Density'
    ),
    yaxis=dict(
        range=[5, 105] 
        ),
    xaxis=dict(
        tickmode='array',
        tickvals=bins) 
)
fig.update_traces(textfont_size=15)

fig.show()
```
![Age group with the highest spending score](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/21fb6f7d-3e40-4b21-94fb-729bc99e145f)

The heatmap suggests that there is no strong correlation between age and spending score, as both high and low spending scores are distributed fairly evenly across different age groups. The density values also do not form any significant patterns that would indicate a clear influence of age on spending. Age 40-41, has the highest count(91) at the highest spending score(100) while Age (64-65) has the highest count at the lowest spending score(5-9).

#### Question 2
Is there a significant difference in spending scores between genders?
```sql
fig = px.density_heatmap(df, x='Gender', y='SpendingScore', 
                         title='Spending Score by Age', color_continuous_scale='blues',
                         text_auto=True)

fig.update_layout(
    title={
        'text': 'Spending Score by Gender',
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_title='Gender',
    yaxis_title='Spending Score',
    font=dict(
        family="Times New Roman",
        size=20,
        color="black"
    ),
    coloraxis_colorbar=dict(
        title='Density'
    ),
    yaxis=dict(
        range=[10, 109]
    )
)
fig.update_traces(textfont_size=15)

fig.show()
```
![Spending score between gender](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/49a840ac-5c90-4eae-88aa-57e9c6ec7ba5)


The highest density observed for female at a spending score of 100 with a density value of 1036. The highest density for males at a spending score of 100 is 972, slightly lower than that of females.

#### Let calculate the P-value.
```sql
# This separate the data into two groups based on gender
male_spending = df[df['Gender'] == 'Male']['SpendingScore']
female_spending = df[df['Gender'] == 'Female']['SpendingScore']

# This perform an independent t-test
t_stat, p_value = ttest_ind(male_spending, female_spending)

print(f'T-statistic: {t_stat:.3f}')
print(f'P-value: {p_value:.3f}')

# This interpret the results
if p_value < 0.05:
    print("There is a significant difference in spending scores between genders.")
else:
    print("There is no significant difference in spending scores between genders.")
```
![P-value](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/306446d3-5ee6-4d48-8d5a-2bfbbf56d527)

Given that the p-value(0.572) is much greater than the common significance level of 0.05 the null hypothesis cannot be rejected.This leads to the conclusion that there is no significant difference in spending score between genders. 

#### Question 3
What's the average annual income of customer buying different product categories? 
```sql
# This group by 'ProductCategory' and calculate the mean of 'AnnualIncome'
average_income_by_category = df.groupby('ProductCategory')['AnnualIncome'].mean().reset_index()
```
```sql
# This rename columns for clarity
average_income_by_category.columns = ['Product Category', 'Average Annual Income']
```
```sql
# This display the results
print(average_income_by_category)

def format_number(value):
    return f'${value / 1000:.2f}k'
```
```sql
# This create a new column in the DataFrame for the formatted text
average_income_by_category['Formatted_Income'] = average_income_by_category['Average Annual Income'].apply(format_number)

fig = px.bar(
    average_income_by_category, 
    x='Product Category', 
    y='Average Annual Income', 
    title='Average Annual Income by Product Category',
    text='Formatted_Income'
)

fig.update_layout(
    title={
        'text': 'Average Annual Income by Product Category',
        'x': 0.5,   # Center the title
        'font': {'family': 'Times New Roman','size': 20}
         },
    xaxis=dict(title=dict(
            font=dict(family="Times New Roman",size=20)
                  ),
        tickfont=dict(family="Times New Roman",size=15)
             ),
    yaxis=dict(
        title=dict(
            font=dict(family="Times New Roman",size=20)
                  ),
        tickfont=dict(family="Times New Roman",size=15)
    )
)

# This update the traces to set the text inside the bars and adjust the text font
fig.update_traces(
    texttemplate='%{text}',  
    textposition='outside',  
    insidetextfont=dict(family='Times New Roman',size=14, color='white'))

fig.show()
```
![Annual average income of customer buying different product category](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/81ee3c40-a578-4338-acf8-eaff3bf73257)

The average annual income for these categories are very close to each other, ranging from approximately 84.82k to 85.31k dollars. The Electronic category has the highest average annual income at 85.31k dollars, The Groceries category has the lowest average annual income at 84.82k dollars. The difference between the highest and lowest average annualincomes is quite small, around 490 dollars.

### Objection 2
#### Question 1
Which product category has the highest sales volume?
```sql
# This count the frequency of each product category
category_counts = df['ProductCategory'].value_counts()

# This get the category with the highest frequency
highest_sales_category = category_counts.idxmax()

print("Category with the highest sales volume:", highest_sales_category)

# This format the numbers with commas
formatted_values = [f"<span style='font-family: Times New Roman'>{value:,}</span>" 
                    for value in category_counts.values]  

fig = go.Figure(data=[go.Bar(x=category_counts.index, 
                             y=category_counts.values, 
                             marker_color='green',
                              text=formatted_values,# Add formatted text labels
                             textposition='auto')])

# This update layout with font settings for title, x-axis, y-axis, and ticks
fig.update_layout(title='Frequency of Product Categories',
                  title_font=dict(family="Times New Roman", size=20, color="black"),
                  title_x=0.5,  # Center the title
                  xaxis=dict(title='Product Category', title_font=dict(family="Times New Roman", 
                     size=16, color="black"), tickangle=-0, tickfont=dict(family="Times New Roman", size=14, color="black")),
                  yaxis=dict(title='Frequency', title_font=dict(family="Times New Roman", size=16,
                             color="black"), tickfont=dict(family="Times New Roman", size=14, color="black")),
                  hovermode='x')
```
![Product category with the highest sales volume](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/cddcee95-e12c-4229-96f9-fd47e6f343ac)

The fequency counts for these categories are very close to each other, with values ranging from approximately 33,172 to 33,479.
The Books category has the highest frequency count at 33,479. The Health & Beauty category has the lowest frequency count at 33,172.

#### Question 2
Which product category generates the highest profit margins?
```sql
# This calculate profit margin for each product
df['ProfitMargin'] = df['Profit'] / df['ProductPrice']

# This group by product category and calculate average profit margin
avg_profit_margin_by_category = df.groupby('ProductCategory')['ProfitMargin'].mean()

# This find the category with the highest average profit margin
highest_profit_category = avg_profit_margin_by_category.idxmax()
highest_avg_profit_margin = avg_profit_margin_by_category.max()

print("Product category with the highest average profit margin:", highest_profit_category)
print("Highest average profit margin:", highest_avg_profit_margin)

# This group by product category and calculate average profit margin
avg_profit_margin_by_category = df.groupby('ProductCategory')['ProfitMargin'].mean().reset_index()

fig = px.bar(
    avg_profit_margin_by_category,
    x='ProductCategory',
    y='ProfitMargin',
    title='Average Profit Margin by Product Category',
    labels={'ProfitMargin': 'Average Profit Margin', 'ProductCategory': 'Product Category'},
    template='plotly_white',
    text='ProfitMargin'
)

fig.update_layout(
    font=dict(family="Times New Roman",size=14),
    title=dict(text='Average Profit Margin by Product Category',font=dict(family="Times New Roman",size=20),
        x=0.5,xanchor='center'))

# This update text template to format values to 2 decimal points
fig.update_traces(texttemplate='%{text:.3f}', textposition='inside')

fig.show()
```
![Product category with highest profit margin](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/555af37f-8d93-467d-83ea-be2704483333)

The average profit margin for each category is nearly similar ranging between 0.073 and 0.076. Books have the highest average profit margin at 0.076, Electronics have the lowest average profit margin at 0.074.

#### Question 3
What is the average product price and how does it correlate with profit?
```sql
# This calculate average product price
average_product_price = df['ProductPrice'].mean()

# This format average product price with a dollar sign
average_product_price_formatted = "${:,.2f}".format(average_product_price)

# This calculate correlation coefficient between product price and profit
correlation_coefficient = df['ProductPrice'].corr(df['Profit'])

print("Average product price:", average_product_price_formatted)
print("Correlation coefficient between product price and profit:", correlation_coefficient)
```
![Average product price and how it correlate with profit](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/07b8b9f6-08ec-4dd4-a659-9e25dc2aa633)

The average product price is 504.59 dollars.
The positive value 0.352 suggests that as the product price increases, the profit also tends to increase. However the relationship is not very strong.
A correlation coefficient of 0.352 is considered to be moderate correlation. This means that there is some degree of linear relationship between product price and profit, but it is not strong enough to predict one variable precisely based on the other.

### Objective 3
#### Question 1
Which shop location has the highest foot traffic?
```sql
# This group by store location and calculate total foot traffic
foot_traffic_by_location = df.groupby('StoreLocation')['FootTraffic'].sum().reset_index()

# This get the minimum foot traffic value
min_foot_traffic = foot_traffic_by_location['FootTraffic'].min()

# Plotting
fig = px.bar(foot_traffic_by_location, x='StoreLocation', y='FootTraffic', 
             title='Total Foot Traffic by Store Location',
             labels={'FootTraffic': 'Total Foot Traffic', 'StoreLocation': 'Store Location'},
             color='FootTraffic',
             color_continuous_scale='Rainbow')

# This set y-axis range to start from the minimum value
fig.update_yaxes(range=[min_foot_traffic, None])
# This update title font and alignment
fig.update_layout(title={'text': 'Total Foot Traffic by Store Location',
                         'x':0.5, 'xanchor': 'center',
                         'font': {'size': 20, 'family': 'Times New Roman'}})

# This update x-axis and y-axis font
fig.update_xaxes(title_text='Store Location', 
                 title_font=dict(family='Times New Roman', size=16),
                 tickfont=dict(family='Times New Roman', size=14))
fig.update_yaxes(title_text='Total Foot Traffic',
                 title_font=dict(family='Times New Roman', size=16),
                 tickfont=dict(family='Times New Roman', size=14))

# This add text labels on top of each bar with foot traffic value
for i, row in foot_traffic_by_location.iterrows():
    # This format foot traffic value with commas
    foot_traffic_formatted = '{:,}'.format(row['FootTraffic'])
    fig.add_annotation(x=row['StoreLocation'],
                       y=row['FootTraffic'],
                       text=foot_traffic_formatted,  # Use formatted value as text
                       font=dict(family='Times New Roman', size=13),
                       showarrow=False,
                       xanchor='center',
                       yanchor='bottom')

fig.show()
```
![Shop location has the highest foot traffic](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/4020e7a9-dbb7-4912-ba59-b2be29a2f12f)

Phoenix is having the highest foot traffic follow by Los Angeles while chicago has the lowest foot Traffic.

#### Question 2
Is there a correlation between foot traffic and profit?
```sql
correlation = df['FootTraffic'].corr(df['Profit'])
print("Correlation coefficient between foot traffic and profit:", correlation)
```
![correlation between foot traffic and profit](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/cda88aad-0356-4ea2-9633-9b0281a6deed)

The correlation value is very close to zero suggesting that there is almost no linear relationship between foot traffic and profit. The slight negative sign implies a very tiny inverse relationship but it is so small that it can be considered statistically insignificant.

#### Question 3
Which store location has the highest average profit per sale?
```sql
# This calculate total profit per store location
total_profit_per_location = df.groupby('StoreLocation')['Profit'].sum()

# This calculate number of sales per store location
number_of_sales_per_location = df.groupby('StoreLocation').size()

# This calculate average profit per sale
average_profit_per_sale = total_profit_per_location / number_of_sales_per_location

# This find the store location with the highest average profit per sale
highest_avg_profit_location = average_profit_per_sale.idxmax()
highest_avg_profit = average_profit_per_sale.max()

print("Store location with the highest average profit per sale:", highest_avg_profit_location)
print("Highest average profit per sale:", highest_avg_profit)

# This prepare data for plotting
avg_profit_df = average_profit_per_sale.reset_index()
avg_profit_df.columns = ['Store Location', 'Average Profit Per Sale']

# Plotting
fig = px.bar(
    avg_profit_df,
    x='Store Location',
    y='Average Profit Per Sale',
    title='Average Profit per Sale by Store Location',
    text='Average Profit Per Sale', color_discrete_sequence=['purple']
)

# This update layout for Times New Roman font and centered title
fig.update_layout(
    font=dict(family="Times New Roman",size=14),
    title=dict(
        text='Average Profit per Sale by Store Location',
        font=dict(family="Times New Roman",size=20),
        x=0.5,
        xanchor='center')
)

# This update text template to format values to 2 decimal points
fig.update_traces(texttemplate='$%{text:.3f}', textposition='inside')

fig.show()
```
![store location with highest average profit per sale](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/0ad8a577-2905-48d5-823b-0488b3823c09)

Houston has the highest average profit per sale while San Antonio has the lowest.

### Objective 4
#### Question 1
How does applying a discount affect the profit margin?
```sql
# This calculate profit margin
df['ProfitMargin'] = (df['Profit'] / df['ProductPrice']) * 100

# This separate the data into two groups
discounted_sales = df[df['DiscountApplied'] == True]
non_discounted_sales = df[df['DiscountApplied'] == False]

# This calculate average profit margin for each group
average_profit_margin_discounted = discounted_sales['ProfitMargin'].mean()
average_profit_margin_non_discounted = non_discounted_sales['ProfitMargin'].mean()

print("Average profit margin for discounted sales: {:.2f}%".format(average_profit_margin_discounted))
print("Average profit margin for non-discounted sales: {:.2f}%".format(average_profit_margin_non_discounted))

# Visualize the comparison
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['Discounted Sales', 'Non-Discounted Sales'],
    y=[average_profit_margin_discounted, average_profit_margin_non_discounted],
    text=[f'{average_profit_margin_discounted:.2f}%', f'{average_profit_margin_non_discounted:.2f}%'],
    textposition='auto',
    marker_color=['indianred', 'lightseagreen']
))

fig.update_layout(
    title={'text': 'Average Profit Margin: Discounted vs Non-Discounted Sales', 
           'x': 0.5, 'xanchor': 'center', 'font': {'size': 20, 'family': 'Times New Roman'}},
    xaxis_title={'text': 'Sales Type', 'font': {'family': 'Times New Roman', 'size': 16}},
    yaxis_title={'text': 'Average Profit Margin (%)', 'font': {'family': 'Times New Roman', 'size': 16}},
    font=dict(family='Times New Roman', size=14)
)

fig.show()
```
![does applying a discount affect the profit margin](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/c9c780b5-9c80-4ec6-92e4-f33f655e96c3)

There is significant difference between the profit margin of discounted and non-discounted sales, with discounted sales having a considerably higher average profit margin.
The non-discounted sales showing a 0.00% profit margin suggests either an error in the collection of the data or an unsual business strategy where non-discounted items are sold at no profit.

#### Question 2
What is the average discount percent applied across different product categories?
```sql
# This  calculate average discount percent by product category
average_discount_percent = df.groupby('ProductCategory')['DiscountPercent'].mean().reset_index()
average_discount_percent.columns = ['Product Category', 'Average Discount Percent']

# This plot with Plotly
fig = px.bar(average_discount_percent, x='Product Category', y='Average Discount Percent',
             title='Average Discount Percent by Product Category',
             labels={'ProductCategory': 'Product Category', 'Average Discount Percent': 'Average Discount Percent (%)'},
             text=average_discount_percent['Average Discount Percent'].round(2),  color_discrete_sequence=['black'])

# This update layout to use Times New Roman font
fig.update_layout(
    title={'text': 'Average Discount Percent by Product Category', 
           'x': 0.5, 'xanchor': 'center', 'font': {'size': 20, 'family': 'Times New Roman'}},
    xaxis_title={'font': {'family': 'Times New Roman', 'size': 16}},
    yaxis_title={'text': 'Average Discount Percent (%)', 'font': {'family': 'Times New Roman', 'size': 16}},
    font=dict(family='Times New Roman', size=15)
)

# This show the plot
fig.show()
```
![average discount percent applied across different product categories](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/f6eade60-32a0-4ccf-9f76-efc2a3e64def)

Books have the highest percentage of average dicount while Electronics have the lowest 7.37, the average discount ranges from 7.37 to 7.57.

#### Question 3
Do customers tends to buy more when discount are applied?
```sql
# This calculate the average product price for transactions with and without discounts
average_price_discounted = df[df['DiscountApplied'] == True]['ProductPrice'].mean()
average_price_non_discounted = df[df['DiscountApplied'] == False]['ProductPrice'].mean()

print("Average product price for discounted sales: {:.2f}".format(average_price_discounted))
print("Average product price for non-discounted sales: {:.2f}".format(average_price_non_discounted))

# This isualize the comparison using Plotly
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['Discounted Sales', 'Non-Discounted Sales'],
    y=[average_price_discounted, average_price_non_discounted],
    text=[f'${average_price_discounted:.2f}', f'${average_price_non_discounted:.2f}'],
    textposition='auto',
    marker_color=['indianred', 'lightseagreen']
))

fig.update_layout(
    title={'text': 'Average Product Price: Discounted vs Non-Discounted Sales', 
           'x': 0.5, 'xanchor': 'center', 'font': {'size': 20, 'family': 'Times New Roman'}},
    xaxis_title={'text': 'Sales Type', 'font': {'family': 'Times New Roman', 'size': 16}},
    yaxis_title={'text': 'Average Product Price', 'font': {'family': 'Times New Roman', 'size': 16}},
    font=dict(family='Times New Roman', size=15)
)

fig.show()
```
![Do customers tends to buy more when discount are applied](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/c03fec23-607c-459e-8d21-2420763913f2)

The average product price for discounted sales is 503.57 dollars, while for non-discounted sales it's 501.61 dollars.
This indicates only a slight difference in average price between the two categories, this suggests that discounts might not significantly lower average selling price, which could be a point of interest for sales strategy analysis.

### Objective 5
#### Question 1
How do competitor price affect sales volume and profit?
```sql
# This create bins for competitor prices
bins = [0, 200, 400, 600, 800, 1000]
labels = ['0-200', '200-400', '400-600', '600-800', '800-1000']
df['CompetitorPriceRange'] = pd.cut(df['CompetitorPrice'], bins=bins, labels=labels, include_lowest=True)

# This calculate average profit and product price for each competitor price range
avg_profit = df.groupby('CompetitorPriceRange')['Profit'].mean().reset_index()
avg_product_price = df.groupby('CompetitorPriceRange')['ProductPrice'].mean().reset_index()

# This merge the data for visualization
avg_data = pd.merge(avg_profit, avg_product_price, on='CompetitorPriceRange')

# THis visualize the data using Plotly
fig = px.bar(
    avg_data,
    x='CompetitorPriceRange',
    y=['Profit', 'ProductPrice'],
    barmode='group',
    title='Effect of Competitor Prices on Profit and Product Price',
    labels={'CompetitorPriceRange': 'Competitor Price Range', 'value': 'Value'},
    text_auto=True
)

# This update text template to include dollar sign and format values
fig.update_traces(texttemplate='$%{y:.2f}', textposition='outside')

# This update layout to use Times New Roman font
fig.update_layout(
    title={'x': 0.5, 'xanchor': 'center'},
    font=dict(family='Times New Roman', size=14),
    title_font=dict(family='Times New Roman', size=20),
    xaxis_title_font=dict(family='Times New Roman', size=20),
    yaxis_title_font=dict(family='Times New Roman', size=20)
)

fig.show()
```
![do competitor price affect sales volume and profit](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/91695cf4-be0c-401f-953d-6d88d090c2d6)

The profit appears to increase with higher competitor price ranges. The product price shows a similar trend, with the highest average price in the 800-1000 range at 862.93 dollars. Both profit and product price are lowest in the 0-200 range , indicating lower profitability and price in this segment.

#### Question 2
Which product categories are the most sensitive to competitor pricing?
```sql
# This calculate profit margin
df['ProfitMargin'] = (df['Profit'] / df['ProductPrice']) * 100

# This create bins for competitor prices
bins = [0, 200, 400, 600, 800, 1000]
labels = ['0-200', '200-400', '400-600', '600-800', '800-1000']
df['CompetitorPriceRange'] = pd.cut(df['CompetitorPrice'], bins=bins, labels=labels, include_lowest=True)

# This calculate average profit margin for each product category and competitor price range
avg_profit_margin = df.groupby(['ProductCategory', 'CompetitorPriceRange'])['ProfitMargin'].mean().reset_index()

# This pivot the table for heatmap visualization
pivot_table = avg_profit_margin.pivot(index='ProductCategory', columns='CompetitorPriceRange', values='ProfitMargin')

# This visualize the data using a heatmap
fig = px.imshow(pivot_table,
                labels=dict(x="Competitor Price Range", y="Product Category", color="Average Profit Margin (%)"),
                x=labels,
                y=pivot_table.index,
                color_continuous_scale='Viridis',
                title='Sensitivity of Product Categories to Competitor Pricing')

# This annotate the heatmap cells with the profit margin values
fig.update_traces(text=pivot_table.values, texttemplate='%{text:.2f}%', textfont=dict(family="Times New Roman"))

# This update layout to use Times New Roman font
fig.update_layout(title={'x':0.45},
    font=dict(family='Times New Roman', size=15),
    title_font=dict(family='Times New Roman', size=20),
    xaxis_title_font=dict(family='Times New Roman', size=16),
    yaxis_title_font=dict(family='Times New Roman', size=16)
)

# While this show the figure
fig.show()
```
![product categories are more sensitive to competitor pricing](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/6f7f6237-5e9f-4c71-a97e-125ff4d3360e)

Books and Clothing categories show relatively stable profit margins across various Competitor price ranges.
Categories like Elctronics,Groceries, and Home & Kitchen show decreasing trend in profit margins as competitor prices increases. The Health & Beauty category exhibits a peak in mid-price range(200-400), indicating higher sensitivity to competitor pricing within this specific range.

### Bonus
This is not part of the objectives but I see a need for this analysis, which is on **Daily purchases**, **Weekly purchases**, **Monthly purchases**, **Monthly purchases by categories** and **Marketing Expenditure vs Sales Volume**, these can also in knowing days of the weeks, weeks of the months and months of the year with higher sales. Also the **Marketing Expenditure vs Sales Volume** can help businesses allocate their marketing budgets more effectively, optimizing spend to maximize sales and profitability.
```sql
# This convert PurchaseDate to datetime
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])

# This extract month and year from PurchaseDate
df['Month'] = df['PurchaseDate'].dt.month
df['Year'] = df['PurchaseDate'].dt.year
```
```sql
# This aggregate number of purchases by date
daily_purchases = df.groupby('PurchaseDate').size().reset_index(name='NumPurchases')

# This visualize daily purchases
fig_daily = px.line(
    daily_purchases, 
    x='PurchaseDate', 
    y='NumPurchases', 
    title='Daily Purchases',
    labels={'NumPurchases': 'Number of Purchases', 'PurchaseDate': 'Date'},
    markers=True
)

# This update layout for readability
fig_daily.update_layout(title={'x':0.45},
    font=dict(family='Times New Roman', size=16),
    title_font=dict(family='Times New Roman', size=20),
    xaxis_title_font=dict(family='Times New Roman', size=18),
    yaxis_title_font=dict(family='Times New Roman', size=18),
    height=700, 
    width=1000, 
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )
)

# This update trace for marker size and opacity
fig_daily.update_traces(marker=dict(size=12, opacity=0.7))

fig_daily.show()
```
![Daily purchases](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/f627ccb7-5e02-4e03-990e-aeab843082ea)
```sql
# This aggregate number of purchases by month
monthly_purchases = df.groupby(['Year', 'Month']).size().reset_index(name='NumPurchases')

# This aggregate number of purchases by week
weekly_purchases = daily_purchases.resample('W-Mon', on='PurchaseDate').sum().reset_index()

# This visualize weekly purchases
fig_weekly = px.line(weekly_purchases, x='PurchaseDate', y='NumPurchases',
                     title='Weekly Purchases',
                     labels={'NumPurchases': 'Number of Purchases', 'PurchaseDate': 'Week Starting'},
                     markers=True)
fig_weekly.update_layout(title={'x': 0.5},
                        font=dict(family='Times New Roman', size=15),
                         title_font=dict(family='Times New Roman', size=20))

fig_weekly.show()
```
![Weekly purchases](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/727b9f96-e902-42ee-9d86-7b3ffe588b3a)

```sql
# This ensure 'Year' is treated as a categorical variable to avoid fractional values in color scale
monthly_purchases['Year'] = monthly_purchases['Year'].astype(str)

# This visualize monthly purchases
fig_monthly = px.bar(
    monthly_purchases, 
    x='Month', 
    y='NumPurchases', 
    color='Year',
    title='Monthly Purchases',
    labels={'NumPurchases': 'Number of Purchases', 'Month': 'Month'},
    color_discrete_sequence=px.colors.qualitative.Set1
)

# This update layout for readability and aesthetics
fig_monthly.update_layout(title={'x': 0.5},
    font=dict(family='Times New Roman', size=15),
    title_font=dict(family='Times New Roman', size=20),
    xaxis_title_font=dict(family='Times New Roman', size=18),
    yaxis_title_font=dict(family='Times New Roman', size=18),
                          
    legend_title_font=dict(family='Times New Roman', size=16),
    legend=dict(title='Year')
)

fig_monthly.show()
```
![Monthly purchases](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/88483964-e80e-4290-80e4-8fad18fca2e5)
```sql
# This aggregate number of purchases by month and product category
monthly_category_purchases = df.groupby(['Year', 'Month', 'ProductCategory']).size().reset_index(name='NumPurchases')

# This visualize monthly purchases by product category
fig_monthly_category = px.bar(
    monthly_category_purchases, 
    x='Month', 
    y='NumPurchases', 
    color='ProductCategory',
    facet_col='Year',
    title='Monthly Purchases by Product Category',
    labels={'NumPurchases': 'Number of Purchases', 'Month': 'Month'}
)

# This update layout to make bars bigger and improve readability
fig_monthly_category.update_layout(
    title={'x': 0.5},
    font=dict(family='Times New Roman', size=20),
    title_font=dict(family='Times New Roman', size=20),
    width=1200, 
    height=500, 
    bargap=0.1,
    xaxis_title_font=dict(family='Times New Roman', size=18),
    yaxis_title_font=dict(family='Times New Roman', size=18),
    legend_title_font=dict(family='Times New Roman', size=16)
)

fig_monthly_category.show()
```
![Month purchases by product categories](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/4f856764-3d69-42f9-9b1d-dceb7d6571da)
```sql
# This aggregate marketing expenditure and sales volume by month
monthly_marketing_sales = df.groupby(['Year', 'Month']).agg({'MarketingExpenditure': 'sum', 'ProductPrice': 'sum'}).reset_index()


# This visualize marketing expenditure vs sales volume
fig_marketing_sales = go.Figure()

fig_marketing_sales.add_trace(go.Scatter(x=monthly_marketing_sales['Month'], y=monthly_marketing_sales['MarketingExpenditure'],
                                         mode='lines+markers',
                                         name='Marketing Expenditure',
                                         line=dict(color='blue')))

fig_marketing_sales.add_trace(go.Scatter(x=monthly_marketing_sales['Month'], y=monthly_marketing_sales['ProductPrice'],
                                         mode='lines+markers',
                                         name='Sales Volume',
                                         line=dict(color='green')))

fig_marketing_sales.update_layout(title='Marketing Expenditure vs Sales Volume',
                                  xaxis_title='Month',
                                  yaxis_title='Value',
                                  font=dict(family='Times New Roman', size=15),
                                  title_font=dict(family='Times New Roman', size=20))

fig_marketing_sales.show()
```
![Marketing Expenditure vs Sales Volume](https://github.com/TomiwaPro/United-States-Retail-records/assets/160256704/1ec488e3-315b-4f0d-8bac-90915af7a337)
#### Note: All the plots created from these analysis are interaction, the images are just the representation of how the outputs look.
Thank you for reading and checking up to this point. What are your thoughts on this case study? Do you agree or disagree with any parts of it? Please share your suggestions, Questions, corrections, or clarifications [HERE](mailto:tomiwaprofficial@gmail.com).

I am actively seeking data analyst positions. Feel free to [Refer me](https://www.linkedin.com/in/tomiwapro/) or [Contact me](mailto:tomiwaprofficial@gmail.com) if you have relevant opportunities. Thank you!





