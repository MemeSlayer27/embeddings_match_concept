statements = {
    # Economic Policy
    'e1': 'We should increase taxes on high-income earners to fund social programs',
    'e2': 'The wealthy should pay their fair share to support society',
    'e3': 'Tax cuts for the wealthy create jobs and economic growth',
    'e4': 'Government spending on social programs should be reduced',
    'e5': 'A universal basic income is necessary in modern society',
    'e6': 'Corporate tax loopholes should be eliminated',
    'e7': 'Free market capitalism works best with minimal regulation',
    'e8': 'Workers deserve a significantly higher minimum wage',
    'e9': 'Economic inequality is a major threat to society',
    'e10': 'Government should not interfere with market forces',
    'e11': 'Break up large corporations to increase competition',
    'e12': 'Reduce regulations to stimulate economic growth',
    'e13': 'The government should subsidize essential industries to protect jobs',
    'e14': 'Reducing income tax is necessary to stimulate economic growth',
    'e15': 'Corporations should be required to contribute to community development',
    'e16': 'Trade tariffs are essential for protecting domestic industries',
    'e17': 'Public spending on infrastructure boosts the economy',
    'e18': 'The free market should decide the price of goods and services',
    'e19': 'Income equality should be prioritized through wealth redistribution',
    'e20': 'We should move towards a cashless society for economic efficiency',
    
    # Healthcare
    'h1': 'Healthcare should be provided as a universal public service',
    'h2': 'Everyone deserves access to affordable healthcare',
    'h3': 'Private healthcare leads to better quality and innovation',
    'h4': 'Government should stay out of healthcare decisions',
    'h5': 'Mental healthcare should be included in basic coverage',
    'h6': 'Healthcare costs are best controlled by market competition',
    'h7': 'Medical decisions should be between doctor and patient only',
    'h8': 'Preventive care should be free for everyone',
    'h9': 'Healthcare is a fundamental human right',
    'h10': 'Private insurance provides the best healthcare options',
    'h11': 'Medicare should be expanded to cover all ages',
    'h12': 'Healthcare innovation requires market incentives',
    'h13': 'Public health funding should prioritize disease prevention',
    'h14': 'Healthcare services should be primarily privately operated',
    'h15': 'Patients should have the freedom to choose any treatment option',
    'h16': 'Government-negotiated drug prices will reduce healthcare costs',
    'h17': 'Free healthcare reduces individual incentives to stay healthy',
    'h18': 'Healthcare must include comprehensive mental health care services',
    'h19': 'Access to healthcare should depend on ability to pay',
    'h20': 'Private-public healthcare partnerships lead to better care',

    
    # Immigration
    'i1': 'Immigration strengthens our economy and enriches our culture',
    'i2': 'We should welcome skilled workers from other countries',
    'i3': 'Immigration levels should be strictly limited',
    'i4': 'Strong borders are essential for national security',
    'i5': 'Provide pathway to citizenship for all immigrants',
    'i6': 'Merit-based immigration system is most effective',
    'i7': 'Sanctuary cities protect community safety',
    'i8': 'Immigration policy should prioritize national interests',
    'i9': 'Cultural diversity improves our society',
    'i10': 'Illegal immigration threatens our sovereignty',
    'i11': 'Family reunification should guide immigration policy',
    'i12': 'Immigration enforcement needs to be stricter',
    'i13': 'Temporary visas should be easier to obtain for economic migrants',
    'i14': 'Citizenship should be granted only after strict vetting processes',
    'i15': 'Immigrants bring valuable skills and innovation to society',
    'i16': 'Border security must be reinforced with new technology',
    'i17': 'Immigration quotas should reflect economic demand',
    'i18': 'Cultural assimilation should be required for all new immigrants',
    'i19': 'Protecting refugees is an international obligation',
    'i20': 'Open borders would be catastrophic for national stability',

    
    # Environment
    'en1': 'Climate change requires immediate government action',
    'en2': 'We must transition to renewable energy sources',
    'en3': 'Environmental regulations hurt business growth',
    'en4': 'Climate policies should not compromise economic development',
    'en5': 'Carbon emissions should be heavily taxed',
    'en6': 'Nuclear power is essential for clean energy',
    'en7': 'Green technology creates economic opportunities',
    'en8': 'Environmental protection costs jobs',
    'en9': 'Fossil fuels are still necessary for economic growth',
    'en10': 'Renewable energy can power our entire economy',
    'en11': 'Individual action is key to environmental protection',
    'en12': 'Market solutions are best for environmental problems',
    'en13': 'Clean water access must be prioritized over corporate interests',
    'en14': 'Banning single-use plastics will protect the environment',
    'en15': 'Climate policy should include nuclear energy solutions',
    'en16': 'Protecting endangered species should take precedence over industry',
    'en17': 'Carbon trading markets are more effective than direct taxes',
    'en18': 'Deforestation must be strictly prohibited',
    'en19': 'Government investment in green tech drives innovation',
    'en20': 'Individuals should bear the cost of their environmental footprint'
}

# Statement pairs with similarity scores (0-1)
statement_pairs = [
    # Very similar statements (0.8-1.0)
    ('e1', 'e2', 0.9),    # Similar progressive tax views
    ('h1', 'h2', 0.85),   # Similar healthcare views
    ('i1', 'i2', 0.8),    # Similar pro-immigration views
    ('en1', 'en2', 0.85), # Similar climate action views
    ('e8', 'e9', 0.85),   # Progressive economic views
    ('h9', 'h11', 0.9),   # Progressive healthcare views
    ('i5', 'i7', 0.8),    # Progressive immigration views
    ('en5', 'en10', 0.85), # Progressive environmental views
    ('e13', 'e17', 0.85), # Similar views on government role in economy
    ('h13', 'h5', 0.9),   # Similar focus on health prevention and mental health
    ('i15', 'i19', 0.8),  # Similar pro-immigration sentiments
    ('en14', 'en18', 0.85), # Environmental protection measures
    ('e15', 'e19', 0.8),  # Focus on corporate responsibility and equality
    ('h16', 'h11', 0.9),  # Government role in healthcare cost control
    ('i13', 'i17', 0.8),  # Economic-based immigration perspectives
    ('en15', 'en19', 0.85), # Mixed energy-environmental innovation views

    
    # Moderately similar statements (0.5-0.7)
    ('en2', 'en4', 0.6),  # Both about climate but different priorities
    ('i2', 'i3', 0.5),    # Both about immigration control but different emphasis
    ('h2', 'h3', 0.55),   # Both about healthcare quality but different approaches
    ('e6', 'e7', 0.6),    # Mixed economic views
    ('h5', 'h7', 0.65),   # Mixed healthcare views
    ('i6', 'i8', 0.7),    # Mixed immigration views
    ('en6', 'en7', 0.6),  # Mixed environmental views
    ('e14', 'e18', 0.6),  # Focus on taxes and market forces
    ('i14', 'i16', 0.55), # Both involve immigration control
    ('h15', 'h7', 0.6),   # Focus on patient autonomy and freedom
    ('e16', 'e13', 0.65), # Government intervention in economy
    ('h18', 'h20', 0.7),  # Public-private partnerships in healthcare
    ('i16', 'i20', 0.5),  # Security and restriction-oriented immigration views
    ('en15', 'en17', 0.6), # Mixed energy approaches to climate
    
    # Opposing statements (0.0-0.3)
    ('e1', 'e3', 0.2),    # Opposing tax views
    ('h1', 'h4', 0.1),    # Opposing healthcare views
    ('i1', 'i4', 0.15),   # Opposing immigration views
    ('en1', 'en3', 0.1),  # Opposing environmental views
    ('e9', 'e10', 0.15),  # Opposing economic views
    ('h9', 'h10', 0.2),   # Opposing healthcare views
    ('i5', 'i12', 0.1),   # Opposing immigration views
    ('en5', 'en9', 0.15), # Opposing environmental views
    ('e14', 'e19', 0.2),  # Opposing views on taxes and redistribution
    ('h14', 'h1', 0.1),   # Private vs public healthcare
    ('i14', 'i1', 0.15),  # Restriction vs openness in immigration
    ('en3', 'en2', 0.1),  # Opposing environment views on regulation
    ('e18', 'e19', 0.2),  # Market freedom vs redistribution
    ('h19', 'h9', 0.15),  # Ability to pay vs universal access
    ('i20', 'i15', 0.1),  # Open borders vs pro-immigration
    ('en18', 'en9', 0.2), # Deforestation vs fossil fuels
    
    # Cross-topic pairs with ideological alignment
    ('e1', 'h1', 0.7),    # Progressive views on tax and healthcare
    ('e3', 'h3', 0.7),    # Conservative views on tax and healthcare
    ('i3', 'en3', 0.65),  # Conservative views on immigration and environment
    ('e9', 'h9', 0.75),   # Strong progressive alignment
    ('e7', 'h6', 0.7),    # Strong conservative alignment
    ('i10', 'en8', 0.65), # Conservative views across topics
    ('e19', 'h9', 0.75),  # Social equality in economy and healthcare
    ('e18', 'h14', 0.7),  # Market-oriented economy and healthcare
    ('i17', 'en17', 0.65), # Economic and market-driven solutions
    ('e17', 'en19', 0.75), # Public spending on economy and environment
    ('e3', 'h14', 0.7),   # Conservative views on taxes and healthcare
    ('i4', 'en8', 0.65),  # Restrictive views across topics
    
    # Cross-topic pairs with little relation
    ('e1', 'i2', 0.4),    # Tax policy vs immigration - some ideological overlap
    ('h2', 'en1', 0.45),  # Healthcare access vs climate - less related
    ('e4', 'i1', 0.3),    # Government spending vs immigration - different topics
    ('e11', 'h5', 0.35),  # Corporate policy vs healthcare - different topics
    ('i8', 'en6', 0.4),   # Immigration vs energy policy
    ('h7', 'en11', 0.3),   # Healthcare autonomy vs environmental responsibility
    ('e15', 'i13', 0.4),  # Corporate responsibility vs immigration
    ('h12', 'en6', 0.45), # Healthcare innovation vs energy
    ('e20', 'h3', 0.35),  # Cashless society vs private healthcare
    ('i18', 'en18', 0.4), # Assimilation vs environmental restrictions
    ('h16', 'en12', 0.3), # Healthcare cost control vs individual action
    ('e14', 'en1', 0.3)   # Tax reduction vs climate action
]

# Load and process the new statements
new_statements = {
    # ECONOMIC POLICY
    # Progressive
    "econ_prog_strong_2": "Nationalize key industries and implement wealth caps",
    "econ_prog_strong_3": "Break up all large corporations and redistribute wealth",
    "econ_prog_strong_4": "Mandate employee ownership in all major companies",
    "econ_prog_mod_2": "Expand social programs through corporate tax reform",
    "econ_prog_mod_3": "Strengthen unions and mandate profit sharing",
    "econ_prog_mod_4": "Implement guaranteed public sector jobs",
    # Moderate
    "econ_mod_3": "Support both small business and worker protections",
    "econ_mod_4": "Targeted incentives for economic development",
    "econ_mod_5": "Promote public-private partnerships for growth",
    # Conservative
    "econ_cons_mod_2": "Simplify tax code and reduce business regulations",
    "econ_cons_mod_3": "Promote free trade with limited protections",
    "econ_cons_mod_4": "Focus on debt reduction and fiscal restraint",
    "econ_cons_strong_2": "Eliminate most business regulations entirely",
    "econ_cons_strong_3": "Privatize all government services possible",
    "econ_cons_strong_4": "Flat tax rate for all income levels",

    # HEALTHCARE
    # Progressive
    "health_prog_strong_2": "Nationalize all hospitals and medical facilities",
    "health_prog_strong_3": "Free universal mental and dental coverage",
    "health_prog_strong_4": "Government control of pharmaceutical industry",
    "health_prog_mod_2": "Expand Medicare to age 50 and above",
    "health_prog_mod_3": "Universal catastrophic coverage with subsidies",
    "health_prog_mod_4": "Mandatory employer-provided health benefits",
    # Moderate
    "health_mod_3": "Reform drug pricing while preserving innovation",
    "health_mod_4": "Increase healthcare price transparency",
    "health_mod_5": "Promote preventive care and wellness programs",
    # Conservative
    "health_cons_mod_2": "Health savings accounts with tax benefits",
    "health_cons_mod_3": "Interstate insurance competition",
    "health_cons_mod_4": "Tort reform to reduce healthcare costs",
    "health_cons_strong_2": "Cash-only medical practice model",
    "health_cons_strong_3": "Eliminate all healthcare mandates",
    "health_cons_strong_4": "Fully privatize Medicare and Medicaid",

    # CLIMATE/ENVIRONMENT
    # Progressive
    "climate_prog_strong_2": "Ban all fossil fuel extraction immediately",
    "climate_prog_strong_3": "Mandatory transition to plant-based diet",
    "climate_prog_strong_4": "Zero-emission requirements for all industries",
    "climate_prog_mod_2": "Green infrastructure investment program",
    "climate_prog_mod_3": "Phase out gas vehicles by 2030",
    "climate_prog_mod_4": "Mandate solar panels on new construction",
    # Moderate
    "climate_mod_3": "Invest in nuclear and renewable energy",
    "climate_mod_4": "Incentivize corporate sustainability",
    "climate_mod_5": "Support clean energy research and development",
    # Conservative
    "climate_cons_mod_2": "Focus on conservation over regulation",
    "climate_cons_mod_3": "Promote voluntary emissions reduction",
    "climate_cons_mod_4": "Balance energy independence with environment",
    "climate_cons_strong_2": "Expand fossil fuel production",
    "climate_cons_strong_3": "Eliminate EPA regulations",
    "climate_cons_strong_4": "Withdraw from climate agreements",

    # IMMIGRATION
    # Progressive
    "immig_prog_strong_2": "Abolish ICE and border patrol",
    "immig_prog_strong_3": "Grant citizenship to all current residents",
    "immig_prog_strong_4": "Provide full benefits to all immigrants",
    "immig_prog_mod_2": "Expand refugee and asylum programs",
    "immig_prog_mod_3": "Create more legal immigration pathways",
    "immig_prog_mod_4": "Support sanctuary city policies",
    # Moderate
    "immig_mod_3": "Modernize visa system and border security",
    "immig_mod_4": "Guest worker programs with oversight",
    "immig_mod_5": "Skills-based immigration with family unity",
    # Conservative
    "immig_cons_mod_2": "Points-based immigration system",
    "immig_cons_mod_3": "Strengthen visa tracking system",
    "immig_cons_mod_4": "Reform chain migration policies",
    "immig_cons_strong_2": "End birthright citizenship",
    "immig_cons_strong_3": "Deport all undocumented immigrants",
    "immig_cons_strong_4": "Build physical barriers on all borders",

    # SOCIAL ISSUES
    # Progressive
    "social_prog_strong_2": "Mandate diversity quotas in all institutions",
    "social_prog_strong_3": "Reparations for historical injustices",
    "social_prog_strong_4": "Restructure all systems for equity",
    "social_prog_mod_2": "Reform police and justice systems",
    "social_prog_mod_3": "Expand civil rights protections",
    "social_prog_mod_4": "Increase funding for social programs",
    # Moderate
    "social_mod_3": "Promote dialogue across differences",
    "social_mod_4": "Evidence-based social policy reform",
    "social_mod_5": "Balance individual rights and community needs",
    # Conservative
    "social_cons_mod_2": "Protect religious freedom rights",
    "social_cons_mod_3": "Support faith-based initiatives",
    "social_cons_mod_4": "Maintain current social structures",
    "social_cons_strong_2": "Return to traditional family values",
    "social_cons_strong_3": "Limit social change through legislation",
    "social_cons_strong_4": "Promote religious values in policy",

    # GUN POLICY
    # Progressive
    "guns_prog_strong_2": "Mandatory gun buyback programs",
    "guns_prog_strong_3": "Ban private gun ownership",
    "guns_prog_strong_4": "Strict liability for gun manufacturers",
    "guns_prog_mod_2": "Create gun ownership database",
    "guns_prog_mod_3": "Require insurance for gun owners",
    "guns_prog_mod_4": "Ban high-capacity magazines",
    # Moderate
    "guns_mod_3": "Improve mental health screening",
    "guns_mod_4": "Register assault-style weapons",
    "guns_mod_5": "Support responsible gun ownership",
    # Conservative
    "guns_cons_mod_2": "State-level gun policy control",
    "guns_cons_mod_3": "Focus on mental health not guns",
    "guns_cons_mod_4": "Protect concealed carry rights",
    "guns_cons_strong_2": "Allow open carry everywhere",
    "guns_cons_strong_3": "Eliminate waiting periods",
    "guns_cons_strong_4": "Constitutional carry nationwide",

    # EDUCATION
    # Progressive
    "edu_prog_strong_2": "Free universal preschool through PhD",
    "edu_prog_strong_3": "Cancel all student debt",
    "edu_prog_strong_4": "Federalize all education systems",
    "edu_prog_mod_2": "Increase teacher pay significantly",
    "edu_prog_mod_3": "Expand early childhood programs",
    "edu_prog_mod_4": "Fund arts and enrichment programs",
    # Moderate
    "edu_mod_3": "Support vocational training options",
    "edu_mod_4": "Reform standardized testing",
    "edu_mod_5": "Modernize curriculum standards",
    # Conservative
    "edu_cons_mod_2": "Promote charter school expansion",
    "edu_cons_mod_3": "Focus on core academics",
    "edu_cons_mod_4": "Support homeschooling rights",
    "edu_cons_strong_2": "Eliminate Department of Education",
    "edu_cons_strong_3": "End public education funding",
    "edu_cons_strong_4": "Full privatization of schools",

    # FOREIGN POLICY
    # Progressive
    "foreign_prog_strong_2": "Close all foreign military bases",
    "foreign_prog_strong_3": "End all military alliances",
    "foreign_prog_strong_4": "Eliminate nuclear weapons",
    "foreign_prog_mod_2": "Expand international aid programs",
    "foreign_prog_mod_3": "Strengthen UN involvement",
    "foreign_prog_mod_4": "Focus on climate cooperation",
    # Moderate
    "foreign_mod_3": "Support strategic partnerships",
    "foreign_mod_4": "Maintain regional stability",
    "foreign_mod_5": "Promote democratic values abroad",
    # Conservative
    "foreign_cons_mod_2": "Strengthen military alliances",
    "foreign_cons_mod_3": "Increase defense readiness",
    "foreign_cons_mod_4": "Support strategic deterrence",
    "foreign_cons_strong_2": "Double military spending",
    "foreign_cons_strong_3": "Unilateral foreign policy",
    "foreign_cons_strong_4": "Expand nuclear arsenal"
    }  # Your pasted statements dictionary




