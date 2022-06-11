
import numpy as np 


class Thermal_model:

    def __init__(self): 
        """
            solar (float): solar intensity -> Typical value: 1380 W/m2
            humidity (float): rel. humidity
        """
        thermal_data = np.genfromtxt('data/ThermalData.csv', delimiter=';', 
        skip_header=1)

        self.solar = thermal_data[:,0]/2
        self.temperature_outside = thermal_data[:,1]
        self.temperature_inside = 20
        self.humidity_outside = thermal_data[:,2]
        self.humidity_inside = 60

        self.new_mass = 0 # [m3/s]
        self.air_mass_in_car = 3 * 1.225 # 0.003 [m3] * 1.225 [kg/m3]
        self.cabin_dashboard_mass = 1 # [kg] dashboard and seats mass 
        self.ac_balance = 0 # [W] thermal power provided by the AC 

        self.dew_point = 10 # [ºC]

        self.Cp_air = 1000 # [W/(kg K)]
        self.Cp_cabin = 1600 # plastic 

        self.Time_Step = 0


    def thermal_balance_update(self, time_step, New_Mass, AC_Balance):
        '''
            balance of energy in the car 
            updates the inside temperature 

            time_step [s]: current time step
            New_Mass [m3/s]: mass entering the car from outside
            AC_Balance [W]: cool/heat power provided by AC
        '''

        self.Time_Step = time_step # [s]
        self.new_mass = New_Mass # [m3/s]

        AC_heat_limit = 10000 # [W]
        AC_cool_limit = -10000 # [W]

        # HVAC cool/heat power limitation
        if AC_Balance > AC_heat_limit:
            self.ac_balance = AC_heat_limit
        elif AC_Balance < AC_cool_limit:
            self.ac_balance = AC_cool_limit
        else:
            self.ac_balance = AC_Balance # [W]

        windshield_area = 2 # [m2]

        thermalBalance = self.solar[self.Time_Step]*windshield_area +\
             self.ac_balance +\
             self.new_mass*self.Cp_air*(
                 self.temperature_outside[self.Time_Step] - self.temperature_inside
                 )

        self.temperature_inside = self.temperature_inside +\
             thermalBalance/(
                 self.Cp_air*self.air_mass_in_car +\
                      self.Cp_cabin*self.cabin_dashboard_mass
                      )

        return thermalBalance # balance in SI units [W]


    def dew_point_confort(self):

        '''
            Dew Point Temperature calculation
            Magnus-Tetens formula (Sonntag90) 

            online dew point calculation by Hanna Pamuła, PhD:
                https://www.omnicalculator.com/physics/dew-point
            wiki article:
                https://en.wikipedia.org/wiki/Dew_point#Calculating_the_dew_point
            original source:
                (Sonntag90) http://irtfweb.ifa.hawaii.edu/~tcs3/tcs3/Misc/Dewpoint_Calculation_Humidity_Sensor_E.pdf
            
            Dew point	Comfort levels:

            (<10°C)	 a bit dry for some
            (10 - 16°C)	 dry and comfortable
            (16 - 18°C)	 getting sticky
            (18 - 21°C)	 unpleasant, lots of moisture in the air
            (>21°C)	 uncomfortable, oppressive, even dangerous above 24ºC
        '''

        # inside humidity update
        if self.new_mass > 0:
            # big simplification, this is inacurate!
            self.humidity_inside = self.humidity_outside[self.Time_Step]*(self.new_mass/self.air_mass_in_car) +\
                self.humidity_inside*(1 - self.new_mass/self.air_mass_in_car)
        else:
            pass # humidity does not change 

        a = 17.62
        b = 243.12 # [°C]

        alpha = np.log(self.humidity_inside/100) +\
             a*self.temperature_inside/(b+self.temperature_inside)

        self.dew_point = (b*alpha) / (a - alpha)

        '''
            confort calculation
        '''

        if self.dew_point < 10: # (<10°C) a bit dry for some
            confort = 1
        elif self.dew_point >= 10 and self.dew_point < 16: # (10 - 16°C) comfortable
            confort = 2
        elif self.dew_point >= 16 and self.dew_point < 18: # (16 - 18°C) getting sticky
            confort = 1
        elif self.dew_point >= 18 and self.dew_point < 21: # (18 - 21°C) unpleasant
            confort = 0
        elif self.dew_point >= 21: # DANGER
            confort = -1000

        return confort


