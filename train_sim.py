import random
from enum import Enum
import numpy as np
import simpy
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


# SIMULATION PARAMETERS
RANDOM_SEED = 42
MIN_PER_HOUR = 60

BASE_RIDER_ARRIVAL_RATE = 4 # riders arriving per minute

DISRUPTION_ENABLED = False # Set to False to disable disruptions
DISRUPTION_START_TIME = 15  # Hour when disruption begins
DISRUPTION_END_TIME = 17   # Hour when disruption ends
DISRUPTED_STATIONS = ["Masjid Jamek"]  # Manually set affected stations
DISRUPTION_SEVERITY = 0.8   # 0-1, where 1 = complete stoppage

RIDER_PATIENCE_BASE = 30  # Base patience in minutes
RIDER_PATIENCE_VARIATION = 15  # +/- minutes variation in patience
RENEGE_PROBABILITY = 0.7  # Probability a rider will renege when patience expires


EVENT_ENABLED = True
EVENT_STATIONS = ["KLCC"]  # Stations near event venues
EVENT_START = 18 
EVENT_END = 23
EVENT_INTENSITY = 0.8  # 0-1 scale (0.8 = 80% more passengers)
EVENT_PASSENGER_RATIO = 0.7  # % of event passengers that are tourists (event goers)


PRINT_EVENTS = False
SAVE_EVENTS_TO_FILE = False # WARNING: maybe memory heavy?
DO_VISUALIZE = True

TRAIN_INTERVAL_PEAK = 4
TRAIN_INTERVAL_REGULAR = 15

TRAIN_START_TIME = 3
STATION_OPENING_TIME = 6
STATION_CLOSING_TIME = 23.5
MORNING_PEAK_START = 7
MORNING_PEAK_END = 9
EVENING_PEAK_START = 17
EVENING_PEAK_END = 20
TRAIN_WAITING_TIME = 0.5

FOUR_COACH_CAPACITY: int = 300 
TWO_COACH_CAPACITY = 150 

STATION_CAPACITY = 500

SIMULATION_END_TIME = 24


if DISRUPTION_ENABLED:
    version = 'disrupted'
elif EVENT_ENABLED:
    version = 'event'
else:
    version = 'baseline'


# Set up the environment
env = simpy.Environment()
G = nx.Graph()
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# region helpers

def to_hours(minutes: int, is_duration: bool = False) -> str:
    hours = int(minutes // 60)
    remaining_minutes = int(minutes % 60)
    if is_duration:
        return str(f"{hours} hour(s) and {remaining_minutes} min(s)")
    else:
        return str(f"{hours:02}:{remaining_minutes:02}")


def create_line(dataframe: pd.DataFrame) -> list:
    line = []
    for _, data in dataframe.iterrows():
        s = Station(env, 
                    name=data["Name"],
                    hub=data["Hub"],
                    comm=data["Commercial"],
                    resd=data["Residential"],
                    tour=data["Tourism"]
                    )
        line.append(s)

    for i in range(len(line)-1):
        t = dataframe["Time"].loc[dataframe.index[i]]
        G.add_edge(line[i], line[i+1], Time=t)
    
    return line


recorded_events = []
def record_event(event: str):
    if PRINT_EVENTS:
        print(event)
    if SAVE_EVENTS_TO_FILE:
        recorded_events.append(event)


def show_progress(env: simpy.Environment):
    while not PRINT_EVENTS:
        yield env.timeout(1)
        progress = (env.now + 1)/(SIMULATION_END_TIME*MIN_PER_HOUR) * 100
        print(f"Simulation progress: {progress:>6.2f}%", end='\r')

# end region


# region stations
class Station(object):
    rider_count = 0

    def __init__(self, env:simpy.Environment, name: str, hub=0, comm=0, resd=1, tour=0):
        self.name = name
        self.queue = []
        self.type_weights = {
            "Hub": hub,
            "Commercial": comm,
            "Residential": resd,
            "Tourism": tour
        }
        
        G.add_node(self, Hub=hub, Commercial=comm, Residential=resd, Tourism=tour)
        self.action = env.process(self.run())

        self.is_disrupted = False
        self.disruption_severity = 0
        self.disruption_resource = simpy.Resource(env, capacity=1)
        self.event_multiplier = 1.0
        self.is_event_active = False


    def __repr__(self):
        return self.name
    

    def run(self):
        yield env.timeout(STATION_OPENING_TIME * MIN_PER_HOUR)

        while env.now < STATION_CLOSING_TIME * MIN_PER_HOUR:
            yield env.timeout(1)
            mult = 1.0

            time_hrs = env.now//MIN_PER_HOUR
            base_rate = BASE_RIDER_ARRIVAL_RATE

            closing_taper_start = STATION_CLOSING_TIME - 3  
            if time_hrs >= closing_taper_start:
                # Linear decay from 1.0 to 0.0 over the last 2 hours
                taper_factor = 1 - (time_hrs - closing_taper_start) / 2  
                base_rate *= max(0, taper_factor)  # Ensure non-negative

            if (MORNING_PEAK_START <= time_hrs < MORNING_PEAK_END):
                rider_amt = base_rate
                mult += self.type_weights["Residential"]
                mult += self.type_weights["Hub"]

            elif (EVENING_PEAK_START <= time_hrs < EVENING_PEAK_END):
                rider_amt = base_rate
                mult += self.type_weights["Commercial"]
                mult += self.type_weights["Hub"]
            
            else:  # Off-peak
                rider_amt = base_rate * (0.5 if time_hrs < 10 else 0.3)  # Reduced daytime flow
                mult += self.type_weights["Hub"] * 0.5

            if time_hrs >= 10:
                mult += self.type_weights["Tourism"]
            

            rider_amt *= mult
            rider_count = np.random.poisson(rider_amt)

            types = list(self.type_weights.keys())
            weights = np.array(list(self.type_weights.values()))
            probabilities = weights/weights.sum()

            
            if len(self.queue) >= STATION_CAPACITY:
                rider_count = 0
            

            for _ in range(rider_count):
                type = np.random.choice(types, p=probabilities) # select randomly station archetype to consider
                roll = random.randint(1, 100) # select random percentage roll for passenger archetype selection 
                match type:
                    case "Hub":
                        if roll < 60:
                            r = Rider(env, self, type=Rider.RiderType.TOURIST)
                        else:
                            r = Rider(env, self)
                    
                    case "Commercial":
                        if roll < 90:
                            r = Rider(env, self, type=Rider.RiderType.COMMUTER)
                        else:
                            r = Rider(env, self)
                    
                    case "Residential":
                        if roll < 60:
                            r = Rider(env, self, type=Rider.RiderType.COMMUTER)
                        if roll > 95:
                            r = Rider(env, self, type=Rider.RiderType.TOURIST)
                        else:
                            r = Rider(env, self)

                    case "Tourism":
                        if roll < 70:
                            r = Rider(env, self, type=Rider.RiderType.TOURIST)
                        else:
                            r = Rider(env, self)

                    case _:
                        r = Rider(env, self)

            # Apply event multiplier
            if self.is_event_active:
                rider_amt *= self.event_multiplier
                rider_count = np.random.poisson(rider_amt)
                
                # Generate event-specific passengers
                for _ in range(int(rider_count * EVENT_PASSENGER_RATIO)):
                    r = Rider(env, self, type=Rider.RiderType.TOURIST)
                
                for _ in range(int(rider_count * (1 - EVENT_PASSENGER_RATIO))):
                    r = Rider(env, self)

    
    def set_disruption(self, severity: float):
        self.is_disrupted = True
        self.disruption_severity = severity
        record_event(f"{to_hours(env.now)}: DISRUPTION STARTED at {self.name} (severity: {severity})")
        # Start patience timers for all riders currently in queue
        for rider in self.queue:
            if not hasattr(rider, 'action') or rider.action.triggered:
                rider.action = env.process(rider.wait_patiently())

    def clear_disruption(self):
        self.is_disrupted = False
        self.disruption_severity = 0
        record_event(f"{to_hours(env.now)}: DISRUPTION ENDED at {self.name}")

    def activate_event(self, intensity: float):
        self.is_event_active = True
        self.event_multiplier = 1 + intensity
        record_event(f"{to_hours(env.now)}: EVENT STARTED at {self.name}")
        
    def deactivate_event(self):
        self.is_event_active = False
        self.event_multiplier = 1.0
        record_event(f"{to_hours(env.now)}: EVENT ENDED at {self.name}")

# endregion

# region riders
# TODO: rider types - regular commuter, occasional commuter, tourist, event goer
class Rider(object):
    class RiderType(Enum):
        TRAVELER = 1,
        COMMUTER = 2,
        TOURIST = 3
    

    def __init__(self, env: simpy.Environment, start: Station, name="", type=RiderType.TRAVELER):
        Station.rider_count += 1
        self.env = env
        self.start = start
        self.name = f"Rider {Station.rider_count}" if name == "" else name
        self.type = type
        self.patience = RIDER_PATIENCE_BASE + random.uniform(-RIDER_PATIENCE_VARIATION, RIDER_PATIENCE_VARIATION)
        self.arrival_time = env.now
        self.has_reneged = False

        self.stop = self.select_destination()
        self.path = nx.shortest_path(G, self.start, self.stop, weight='weight')
        self.start_time = env.now
        self.embark_times = []
        self.disembark_times = []
        self.total_ride_time = 0
        self.end_time = 0
        self.train_count = -1
        record_event(f"{to_hours(env.now)}: {self.name} arrived at {self.start} en route to {self.stop}")
        start.queue.append(self)

        # Start patience timer if station is disrupted
        if start.is_disrupted:
            self.action = env.process(self.wait_patiently())
    

    def select_destination(self) -> Station:
        eligible_stations = {}
        time_hrs = env.now//MIN_PER_HOUR

        match self.type:
            case Rider.RiderType.COMMUTER:
                if (MORNING_PEAK_START <= time_hrs < MORNING_PEAK_END):
                    eligible_stations = {station: weights for station, weights in G.nodes.data("Commercial") if station != self.start}
                elif (EVENING_PEAK_START <= time_hrs < EVENING_PEAK_END):
                    eligible_stations = {station: weights for station, weights in G.nodes.data("Residential") if station != self.start}
                else:
                    # assigns all stations with equal weightage
                    eligible_stations = {station: 1 for station in G.nodes() if station != self.start}

            case Rider.RiderType.TOURIST:
                eligible_stations = {station: weights for station, weights in G.nodes.data("Tourism") if station != self.start}
                hubs = {station: weights*0.5 for station, weights in G.nodes.data("Hub") if station != self.start}
                eligible_stations.update(hubs)

                # If during event hours, boost tourism weights
                if EVENT_START <= env.now//MIN_PER_HOUR < EVENT_END:
                    for station in eligible_stations:
                        if station.name in EVENT_STATIONS:
                            eligible_stations[station] *= 2  # Double attraction during events
                
            case _:
                eligible_stations = {station: 1 for station in G.nodes() if station != self.start}
                if EVENT_START <= env.now//MIN_PER_HOUR < EVENT_END:
                    for station in eligible_stations:
                        if station.name in EVENT_STATIONS:
                            eligible_stations[station] *= 1.2  # Double attraction during events
            
        # Apply duration decay (inverse square law)
        # Use time as weights, the longer a journey takes, the less a rider will likely to choose it 
        duration_weights = {}
        for station in eligible_stations:
            path = nx.shortest_path(G, self.start, station, weight='Time')
            travel_time = sum(G.edges[path[i], path[i+1]]['Time'] for i in range(len(path)-1))
            duration_weights[station] = 1 / (travel_time ** 2 + 1)  # +1 to avoid division by zero

        # Combine type weights and duration weights
        combined_weights = {
            station: eligible_stations[station] * duration_weights[station] * 1000  # Scale up
            for station in eligible_stations
        }

        # Normalize probabilities
        total_weight = sum(combined_weights.values())
        probabilities = {station: w/total_weight for station, w in combined_weights.items()}
        
        stations = list(probabilities.keys())
        probs = list(probabilities.values())
        
        return np.random.choice(stations, p=probs)
    
    
    def embark(self, at: Station):
        self.train_count += 1
        self.embark_times.append(env.now)
        at.queue.remove(self)
        record_event(f"{to_hours(env.now)}: {self.name} embarked at {at}")
        
    

    def disembark(self, at: Station):
        self.disembark_times.append(env.now)
        record_event(f"{to_hours(env.now)}: {self.name} disembarked at {at}")

        if at == self.stop:
            self.end_time = env.now
            for i in range(self.train_count+1):
                self.total_ride_time = self.disembark_times[i] - self.embark_times[i]
            record_event(f"       {self.name} rode from {self.start} to {self.stop} for {to_hours(self.total_ride_time, True)}")
        else:
            at.queue.append(self)
    

    def wait_patiently(self):
        yield self.env.timeout(self.patience)
        
        # Only consider reneging if still in queue and station is still disrupted
        if (self in self.start.queue and 
            self.start.is_disrupted and 
            random.random() < RENEGE_PROBABILITY):
            
            self.renege()
    

    def renege(self):
        # Rider gives up waiting and leaves the system
        if self in self.start.queue:
            self.start.queue.remove(self)
            self.has_reneged = True
            wait_time = (self.env.now - self.arrival_time) / MIN_PER_HOUR
            record_event(f"{to_hours(self.env.now)}: {self.name} reneged after waiting {wait_time:.1f} hours at {self.start}")

        
# endregion

# region trains
class Train(object):
    all_trains = {}

    def __init__(self, env: simpy.Environment, path: list, limit: int):
        self.env = env
        self.route = path.copy()
        self.limit = limit
        self.passengers = []
        self.current_position = 0  # 0 to 1 representing progress between stations
        self.current_edge = (self.route[0], self.route[1])
        self.action = env.process(self.run())
        Train.all_trains[id(self)] = self
        
    
    def run(self):
        current_station: Station = self.route.pop(0)
        next_station: Station = self.route.pop(0)  
        self.current_edge = (current_station, next_station)
        
        while next_station is not None:
            if current_station.is_disrupted:
                delay = random.uniform(5, 15) * current_station.disruption_severity
                record_event(f"{to_hours(env.now)}: Train delayed at {current_station} for {delay:.1f} mins")
                yield env.timeout(delay)

            # Board passengers (normal operation)
            for q in current_station.queue[:]:
                if q.has_reneged:
                    continue
                if len(q.path) > 1 and q.path[1] == next_station:
                    if len(self.passengers) < self.limit:
                        self.passengers.append(q)
                        q.embark(current_station)
            
            yield env.timeout(TRAIN_WAITING_TIME)

            # Handle track disruption between stations
            if current_station.is_disrupted or next_station.is_disrupted:
                edge_time = G.edges[current_station, next_station]["Time"] 
                disruption_delay = edge_time * max(current_station.disruption_severity, 
                                                 next_station.disruption_severity)
                record_event(f"{to_hours(env.now)}: Train slowed between {current_station} and {next_station}")
                yield env.timeout(edge_time + disruption_delay)
            else:
                # Normal travel

                # Travel to next station
                edge_time = G.edges[current_station, next_station]["Time"]
                yield env.timeout(edge_time)
            
                # Arrive at next station
                current_station = next_station
                next_station = self.route.pop(0) if len(self.route) > 0 else None
                
                if next_station:
                    self.current_edge = (current_station, next_station)
                
                # Disembark passengers
                for p in self.passengers[:]:
                    if len(p.path) > 1:
                        p.path.pop(0)
                    if len(p.path) == 1 or (len(p.path) > 1 and not p.path[1] == next_station):
                        self.passengers.remove(p)
                        p.disembark(current_station)
        
        Train.all_trains.pop(id(self))


def generate_trains(env: simpy.Environment):
    yield env.timeout(TRAIN_START_TIME * MIN_PER_HOUR)
    while env.now < STATION_CLOSING_TIME * MIN_PER_HOUR:
        time_hrs = env.now / MIN_PER_HOUR

        is_peak = (MORNING_PEAK_START <= time_hrs < MORNING_PEAK_END) or \
                 (EVENING_PEAK_START <= time_hrs < EVENING_PEAK_END)

        # Determine interval and train type
        if is_peak:
            inter = TRAIN_INTERVAL_PEAK
            # During peak, use mix of 2-coach and 4-coach trains
            if random.random() < 0.3:  # 30% chance of 2-coach
                capacity = TWO_COACH_CAPACITY
            else:
                capacity = FOUR_COACH_CAPACITY
        else:
            inter = TRAIN_INTERVAL_REGULAR
            capacity = FOUR_COACH_CAPACITY
        
        
        # Launch trains in both directions
        Train(env, kj_line, capacity)
        Train(env, list(reversed(kj_line)), capacity)

        yield env.timeout(inter)

# endregion

# region disruptions
# DISRUPTION PARAMETERS
def manage_disruptions(env):      
    # Wait for disruption start time
    yield env.timeout(DISRUPTION_START_TIME * MIN_PER_HOUR)
    
    # Apply disruptions
    for station in G.nodes():
        if station.name in DISRUPTED_STATIONS:
            station.set_disruption(DISRUPTION_SEVERITY)
    
    # Wait for disruption duration
    yield env.timeout((DISRUPTION_END_TIME - DISRUPTION_START_TIME) * MIN_PER_HOUR)
    
    # Clear disruptions
    for station in G.nodes():
        if station.name in DISRUPTED_STATIONS:
            station.clear_disruption()


def manage_events(env):
    if not EVENT_ENABLED:
        return
        
    # Wait for event start time
    yield env.timeout(EVENT_START * MIN_PER_HOUR)
    
    # Activate events at specified stations
    for station in G.nodes():
        if station.name in EVENT_STATIONS:
            station.activate_event(EVENT_INTENSITY)
    
    # Wait for event duration
    yield env.timeout((EVENT_END - EVENT_START) * MIN_PER_HOUR)
    
    # Deactivate events
    for station in G.nodes():
        if station.name in EVENT_STATIONS:
            station.deactivate_event()

# endregion


# region statistics
STATS_COLLECTION_INTERVAL = 30  # minutes
station_stats = []  # To store passenger counts over time
train_load_stats = []  # To store train passenger counts

def collect_statistics(env):
    while env.now <= SIMULATION_END_TIME*MIN_PER_HOUR:
        # Record passenger counts at stations
        time_hrs = env.now / MIN_PER_HOUR
        station_data = {
            'time': time_hrs,
            'passenger_count': {station.name: len(station.queue) for station in G.nodes()}
        }
        station_stats.append(station_data)
        
        # Record train loads
        for train_id, train in Train.all_trains.items():
            train_load_stats.append({
                'time': time_hrs,
                'passenger_count': len(train.passengers),
                'load_percentage': (len(train.passengers)/train.limit)*100
            })
        
        yield env.timeout(STATS_COLLECTION_INTERVAL)

def plot_passenger_counts():
    # Convert to DataFrame for easier manipulation
    times = [entry['time'] for entry in station_stats]
    stations = list(station_stats[0]['passenger_count'].keys())
    
    # Create a DataFrame with time as index and stations as columns
    data = {}
    for station in stations:
        data[station] = [entry['passenger_count'][station] for entry in station_stats]
    
    df = pd.DataFrame(data, index=times)
    
    # Plot

    cmap = plt.cm.get_cmap('hsv', len(stations))

    plt.figure(figsize=(12, 10))
    for i, station in enumerate(stations):
        plt.plot(df.index, df[station], color=cmap(i), label=station)
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Passenger Count')
    plt.title('Passenger Counts at Stations Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(5, 25)
    plt.savefig(f'passenger_counts({version}).png')
    plt.show()


def plot_combined_passenger_counts():
    times = [entry['time'] for entry in station_stats]
    
    # Calculate totals
    station_totals = [sum(entry['passenger_count'].values()) for entry in station_stats]
    
    # Get train passenger totals
    train_totals = []
    for t in times:
        relevant = [x for x in train_load_stats if x['time'] == t]
        train_totals.append(sum(x['passenger_count'] for x in relevant))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, station_totals, label='Passengers at Stations', linewidth=2)
    plt.plot(times, train_totals, label='Passengers on Trains', linewidth=2)
    
    # Formatting
    plt.xlabel('Time (hours)')
    plt.ylabel('Passenger Count')
    plt.title('Total System Passenger Load')
    plt.legend()
    plt.grid(True)
    
    # Highlight peak periods
    plt.axvspan(MORNING_PEAK_START, MORNING_PEAK_END, color='yellow', alpha=0.2, label='Peak Hours')
    plt.axvspan(EVENING_PEAK_START, EVENING_PEAK_END, color='yellow', alpha=0.2)
    
    plt.tight_layout()
    plt.xlim(5, 25)
    plt.savefig(f'combined_passenger_counts ({version}).png')
    plt.show()


def plot_train_utilization():
    df = pd.DataFrame(train_load_stats)
    
    # Calculate average utilization per time period
    df['time_bin'] = pd.cut(df['time'], bins=np.arange(6, 25, 1))
    utilization = df.groupby('time_bin', observed=True)['load_percentage'].max()
    
    # Plot
    plt.figure(figsize=(12, 6))
    utilization.plot(kind='bar')
    plt.xlabel('Time of Day')
    plt.ylabel('Average Train Load (%)')
    plt.title('Train Utilization Throughout the Day')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'train_utilization({version}).png')
    plt.show()


def plot_peak_hour_queues():
    # Get data for peak hours
    morning_peak = [entry for entry in station_stats 
                   if MORNING_PEAK_START <= entry['time'] < MORNING_PEAK_END]
    evening_peak = [entry for entry in station_stats 
                   if EVENING_PEAK_START <= entry['time'] < EVENING_PEAK_END]
    
    # Calculate average queue lengths
    stations = list(station_stats[0]['passenger_count'].keys())
    morning_avgs = {station: np.mean([entry['passenger_count'][station] for entry in morning_peak]) 
                   for station in stations}
    evening_avgs = {station: np.mean([entry['passenger_count'][station] for entry in evening_peak]) 
                   for station in stations}
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Morning peak
    ax1.bar(morning_avgs.keys(), morning_avgs.values())
    ax1.set_title(f'Average Queue Lengths ({MORNING_PEAK_START}:00-{MORNING_PEAK_END}:00)')
    ax1.set_ylabel('Passenger Count')
    ax1.tick_params(axis='x', rotation=90)
    
    # Evening peak
    ax2.bar(evening_avgs.keys(), evening_avgs.values())
    ax2.set_title(f'Average Queue Lengths ({EVENING_PEAK_START}:00-{EVENING_PEAK_END}:00)')
    ax2.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.savefig(f'peak_hour_queues({version}).png')
    plt.show()

# endregion



# region runtime

# Generate line through CSV file
pd.options.display.max_rows = 50

kj_df = pd.read_csv("kelana_jaya_data.csv")
kj_line = create_line(kj_df)

# Start the simulation

env.process(generate_trains(env))
env.process(show_progress(env))

if DISRUPTION_ENABLED:
    env.process(manage_disruptions(env))
if EVENT_ENABLED:
    env.process(manage_events(env))

if DO_VISUALIZE:
    env.process(collect_statistics(env))


print("Starting simulation...")
env.run(SIMULATION_END_TIME*MIN_PER_HOUR)  # Run for 24 hours
print("")


if SAVE_EVENTS_TO_FILE:
    with open("train_records.txt", 'w') as f:
        for event in recorded_events:
            f.write(event + "\n")
    
    
if DO_VISUALIZE:
    print("Generating static plots...")
    plot_passenger_counts()
    plot_combined_passenger_counts()
    plot_train_utilization()
    plot_peak_hour_queues()
    print("Static plots saved as PNG files.")

else:
    print("Simulation complete.")