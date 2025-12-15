"""Configuration constants for the Real Estate Dashboard.

This module contains all configuration constants including state mappings,
dataset labels, and formatting configurations used throughout the dashboard.
"""

# State Selection
STATES = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'IA': 'Iowa',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts',
    'MD': 'Maryland',
    'ME': 'Maine',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MO': 'Missouri',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'NE': 'Nebraska',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NV': 'Nevada',
    'NY': 'New York',
    'OH': 'Ohio',
    'OK': 'Oklahoma', 
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia',
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming'
}

# dataset labels
DATASET_LABELS = {
    'median_days_on_market': 'Median Days on Market',
    'average_listing_price': 'Average Listing Price',
    'median_listing_price': 'Median Listing Price',
    'median_listing_price_per_sqft': 'Median Listing Price per Square Foot',
    'active_listing_count': 'Active Listings',
    'pending_listing_count': 'Pending Listings',
    'new_listing_count': 'New Listings',
    'price_decrease_count': 'Price Decreases',
    'price_decrease_ratio': 'Price Decrease Ratio',
    'price_increase_count': 'Price Increases',
    'price_increase_ratio': 'Price Increase Ratio',
    'supply_score': 'Supply Score',
    'demand_score': 'Demand Score',
    'median_square_feet': 'Median Square Feet'
}

# dataset titles
DATASET_TITLES = {
    'median_days_on_market': 'Median Days on Market',
    'average_listing_price': 'Average Listing Price',
    'median_listing_price': 'Median Listing Price',
    'median_listing_price_per_sqft': 'Median Listing Price per Square Foot',
    'active_listing_count': 'Active Listings',
    'pending_listing_count': 'Pending Listings',
    'new_listing_count': 'New Listings',
    'price_decrease_count': 'Number of Listings with Price Decreases',
    'price_decrease_ratio': 'Price Decrease Ratio',
    'price_increase_count': 'Number of Listings with Price Increases',
    'price_increase_ratio': 'Price Increase Ratio',
    'supply_score': "Supply Score (Realtor.com's Market Hotness Index)",
    'demand_score': "Demand Score (Realtor.com's Market Hotness Index)",
    'median_square_feet': 'Median Square Feet'
}

# for datasets that need colors flipped (up = bad, down = good)
REVERSED_DATASETS = {
    'median_days_on_market',
    'price_decrease_count',
    'price_decrease_ratio',
    'supply_score'
}

# Datasets that default to metro-only
METRO_ONLY_DATASETS = {
    'price_decrease_count',
    'price_increase_count',
    'price_decrease_ratio',
    'price_increase_ratio',
    'active_listing_count',
    'new_listing_count',
    'pending_listing_count',
    'supply_score',
    'demand_score',
    'median_listing_price_per_sqft'
}

DATASET_FORMAT_CONFIG = {
    'median_days_on_market': {'prefix': '', 'suffix': ' days', 'format': '.0f'},
    'average_listing_price': {'prefix': '$', 'suffix': '', 'format': ',.0f'},
    'median_listing_price': {'prefix': '$', 'suffix': '', 'format': ',.0f'},
    'median_listing_price_per_sqft': {'prefix': '$', 'suffix': '', 'format': ',.0f'},
    'active_listing_count': {'prefix': '', 'suffix': ' listings', 'format': ',.0f'},
    'new_listing_count': {'prefix': '', 'suffix': ' listings', 'format': ',.0f'},
    'pending_listing_count': {'prefix': '', 'suffix': ' listings', 'format': ',.0f'},
    'price_decrease_count': {'prefix': '', 'suffix': ' listings', 'format': ',.0f'},
    'price_increase_count': {'prefix': '', 'suffix': ' listings', 'format': ',.0f'},
    'price_decrease_ratio': {'prefix': '', 'suffix': '%', 'format': '.1f'},
    'price_increase_ratio': {'prefix': '', 'suffix': '%', 'format': '.1f'},
    'supply_score': {'prefix': '', 'suffix': '', 'format': '.1f'},
    'demand_score': {'prefix': '', 'suffix': '', 'format': '.1f'},
    'median_square_feet': {'prefix': '', 'suffix': 'sqft', 'format': '.0f'}
}

# base monotone colors
MONO_COLORS = [
    'rgb(8,48,107)','rgb(8,81,156)','rgb(33,113,181)',
    'rgb(66,146,198)', 'rgb(107,174,214)','rgb(198,219,239)',
    'rgb(158,202,225)','rgb(222,235,247)', 'rgb(247,251,255)'
]



# Time Series Chart Color
TS_COLORS = [
    "rgb(62,0,179)",     # Interdimensional Blue (#3E00B3)
    "rgb(33,0,237)"      # X11 Purple (#2100ED, slightly lighter purple)
]