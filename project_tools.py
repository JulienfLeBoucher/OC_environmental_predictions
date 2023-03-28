### This file was helpful to fly over the EDA while writing down some
# TODO's if i'd want to easily go back to the data cleansing
# and feature engineering.
#
# It helped not to lose too much time and focus on the core of the
# project.

### CONSTANTS
PPT = 'PrimaryPropertyType'
GHG_TARGET = 'TotalGHGEmissions'
GHG_INTENSITY_TARGET = 'GHGEmissionsIntensity'
ENERGY_TARGET = 'SiteEnergyUse(kBtu)'
ENERGY_INTENSITY_TARGET = 'SiteEUI(kBtu/sf)'
TARGETS = [GHG_TARGET, 
           GHG_INTENSITY_TARGET,
           ENERGY_TARGET, 
           ENERGY_INTENSITY_TARGET]

### Cleansing functions
def neighborhood_correction(data):
    """ formatting strings """
    pass

def SiteEnergyUse_normalization_correction(data):
    """ Some values were imputed 0. Need to correct that if used as 
    target."""
    pass

def number_of_buildings_correction(data):
    """ 0 impossible """
    pass

def number_of_floors_correction(data):
    """ 0 impossible """
    pass

def percentage_primary_type_use_threshold_choice(data):
    """  should not be superior to 1"""
    pass

def dtypes_optimization(data):
    pass
    # Categorization
    # categories = ['BuildingType',
    #               'PrimaryPropertyType',
    #               'City',
    #               'State',
    #               'Neighborhood',
    #               'ComplianceStatus']

    # data.loc[:, categories] = (data.loc[:, categories]
    #                            .apply(lambda x: x.astype('category'), axis=0))
    
    # Integers    
    return data
### End of dtypes_optimization

def selecting_non_residential_buildings(data):
    """ Filtering on BuildingType and PrimaryPropertyType """
    # # BuildingType
    # sel = ['NonResidential',
    #        'Nonresidential COS',
    #        'Campus',               
    #        'SPS-District K-12',    
    #        'Nonresidential WA']
    
    # sel2 = ['Senior Care Community',
    #         'College/University',
    #         'Hotel',
    #         'Residence Hall/Dormitory',
    #         'Parking',
    #         'Office']
    
    # mask = (data.BuildingType.isin(sel)) 
    #         # & (data.LargestPropertyUseType.isin(sel2)))
    
    # data = data.loc[mask, :]
    # # At this point, There are still some properties to discard.
    # # The ones which were not in sel but which have Multifamily Housing
    # # In the Largest Use.
    # data = data.loc[data.LargestPropertyUseType != 'Multifamily Housing']
    return data
    
def selecting_reliable_properties(data):
    """ Avoiding default values, non-compliant and missing data
    in ComplianceStatus"""
    data = data.loc[data.ComplianceStatus == 'Compliant', :]    
    return data

def data_cleansing(data):
    """  """
    data = data.drop('Comments', axis=1) # Hold no info
    selecting_reliable_properties(data)
    selecting_non_residential_buildings(data)    
    return data

### Modeling functions
