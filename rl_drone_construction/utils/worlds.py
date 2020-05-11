import json
import copy
import numpy as np
from multiagent.core import World
from rl_drone_construction.utils.entities import Drone, TargetLandmark, SupplyEntity, SupplyBuffer, ThreatEntity, DockEntity, BrickEntity, SprayEntity

class DroneWorld(World):

    def __init__(self, n_lidar_per_agent, mem_frames, dt=0.1):
        super().__init__()
        from utils.lidar import RayLidar
        self.lidar = RayLidar(self, n_lidar_per_agent=n_lidar_per_agent)
        self.n_lidar_per_agent = n_lidar_per_agent
        self.mem_frames = mem_frames
        self.built = []
        self.unbuilt = []
        self.dt = dt

    @property
    def entities(self):
        return self.agents + self.landmarks + self.built

    @property
    def targets(self):
        return [l for l in self.landmarks if isinstance(l, TargetLandmark)]

    @property
    def threats(self):
        return [l for l in self.landmarks if isinstance(l, ThreatEntity)]

    def step(self):
        self.pre_step()
        super().step()
        self.post_step()

    def pre_step(self):
        for agent in self.agents:
            assert isinstance(agent, Drone)
            agent.previous_state.p_pos = np.copy(agent.state.p_pos)
            agent.previous_state.p_vel = np.copy(agent.state.p_vel)
            n = np.linalg.norm(agent.state.p_vel)
            agent.state.p_rot = agent.state.p_vel / n if n > 0 else np.array(
                [0., 1.])

    def post_step(self):
        for agent in self.agents:
            assert len(agent.lidar_memory) == 2
            agent.lidar_memory.pop(0)
            agent.lidar_memory.append(agent.agents_lidar)
            agent.agents_lidar = self.lidar.get_ray_lidar(agent)


class DroneConstructionWorld(World):

    def __init__(self, n_lidar_per_agent, mem_frames, dt=0.1):
        super().__init__()
        from utils.lidar import RayLidar
        self.lidar = RayLidar(self, n_lidar_per_agent=n_lidar_per_agent)
        self.n_lidar_per_agent = n_lidar_per_agent
        self.mem_frames = mem_frames
        self.js_path = None
        self.js = None
        self.all_targets = tuple()
        self.docks = []
        self.supplies = []
        self.threats = []
        self.dt = dt
        self.buffers = []
        self.unbuilt = []
        self.buildable = []
        self.built = []

    def init_from_json(self, path):
        raise NotImplementedError

    def reset_progress(self):
        self.built = []
        self.unbuilt = list(self.all_targets)
        self.buildable = list(self.all_targets)
        self.buffers = [SupplyBuffer(s.uid, s) for s in self.supplies]

    def load_components(self):
        dt = self.js['world']['dt']
        if self.dt != dt:
            print("change world.dt to", dt)
            self.dt = dt
        docks = self.js['docks']
        for i, dock in enumerate(docks):
            if i < len(self.docks):
                d = self.docks[i]
            else:
                d = DockEntity(uid=dock['id'])
                self.docks.append(d)
            d.state.p_pos = np.array(dock['pos'][:2], dtype=float)
            d.state.p_alt = dock['pos'][2]
            d.state.p_rot = dock['rot']
        self.docks = self.docks[:i + 1]

        supplies = self.js['supplies']
        for i, supply in enumerate(supplies):
            if i < len(self.supplies):
                s = self.supplies[i]
                s.uid = supply['id']
            else:
                s = SupplyEntity(uid=supply['id'], type=supply['type'])
                self.supplies.append(s)
            s.size = self.supplies[0].size
            s.state.p_pos = np.array(supply['pos'][:2], dtype=float)
            s.state.p_alt = supply['pos'][2]
            s.state.p_rot = supply['rot']
        self.supplies = self.supplies[:i + 1]

        threats = self.js['threats']
        for i, threat in enumerate(threats):
            if i < len(self.threats):
                t = self.threats[i]
                t.uid = threat['id']
            else:
                t = ThreatEntity(uid=threat['id'])
                self.threats.append(t)

            t.size = threat['size'] + 0.03  # TODO prevent collision
            t.state.p_pos = np.array(threat['pos'][:2], dtype=float)
            t.state.p_alt = 0
            t.state.p_rot = 0
        self.threats = self.threats[:i + 1]

    @property
    def entities(self):
        return self.agents + self.landmarks + self.built

    @property
    def targets(self):
        return [l for l in self.landmarks if isinstance(l, TargetLandmark)]

    def step(self):
        self.pre_step()
        super().step()
        self.post_step()
        self.update_config()

    def pre_step(self):
        for agent in self.agents:
            agent.previous_state.p_pos = np.copy(agent.state.p_pos)
            agent.previous_state.p_vel = np.copy(agent.state.p_vel)
            if agent.supervised and len(agent.supervise_fn):
                agent.collide = False
                agent.movable = False
                done = agent.supervise_fn[0](*agent.supervise_param[0])
                if done or done is None:
                    agent.supervise_fn.pop(0)
                    agent.supervise_param.pop(0)
                    if len(agent.supervise_fn) == 0:
                        agent.movable = True
                        agent.collide = True
                        agent.supervised = False
            else:
                n = np.linalg.norm(agent.state.p_vel)
                if n > 0:
                    rot = agent.state.p_vel / n
                    agent.state.p_rot = (-np.arctan2(*rot)) % (2 * np.pi)
                else:
                    agent.state.p_rot = 0

    def post_step(self):
        pass

    def update_config(self):
        with open(self.js_path, 'r') as f:
            js = json.load(f)
        if js == self.js:
            return
        self.js = js
        self.load_components()
        while len(self.agents) < len(self.docks):
            i = len(self.agents)
            agent = Drone(uid=i)
            target = TargetLandmark()
            agent.target = target
            agent.dock = self.docks[i]
            agent.state = copy.copy(self.docks[i].state)
            agent.name = 'agent %d' % i
            agent.uid = i
            agent.size = 0.3
            agent.lidar_range = 4.5
            agent.pseudo_collision_range = 0.3
            agent.construct_range = self.supplies[0].size - agent.size
            agent.construct_maxvel = 5
            agent.construct_capacity = 1
            agent.cur_construct_capacity = 0
            agent.battery = 100
            agent.load = None
            agent.target.state = copy.copy(self.supplies[0].state)
            agent.previous_state.p_pos = np.copy(agent.state.p_pos)
            agent.previous_state.p_vel = np.copy(agent.state.p_vel)
            agent.agents_lidar = self.lidar.get_ray_lidar(agent)
            agent.lidar_memory = [agent.agents_lidar, agent.agents_lidar]
            agent.supervised = False
            agent.supervise_fn = []
            agent.supervise_param = []
            agent.supervise(agent.supervise_charge, [self])
            self.agents.append(agent)
            self.landmarks.append(target)

        while len(self.agents) > len(self.docks):
            agent = self.agents[-1]
            self.buildable.append(agent.allocated)
            for supply in self.supplies:
                if supply.cur_agent is agent:
                    supply.cur_agent = None
            for buffer in self.buffers:
                for i in range(len(buffer.queue)):
                    if buffer.queue[i][1] is agent:
                        buffer.queue.pop(i)
            self.agents.pop()
            self.landmarks.pop()

    def gather_stream_data(self):
        raise NotImplementedError


class BrickLayingWorld(DroneConstructionWorld):

    def init_from_json(self, path):
        self.js_path = path
        with open(path, 'r') as f:
            self.js = json.load(f)
        bricks = self.js['bricks']
        all_targets = []
        for row in bricks:
            for brick in row:
                b = BrickEntity(uid=brick['id'],
                                target_pos=np.array(brick['pos']),
                                target_rot=brick['rot'],
                                type=brick['type'])
                all_targets.append(b)
        i = 0
        for row in bricks:
            for brick in row:
                for p in brick['dids']:
                    all_targets[i].register_parent(all_targets[p])
                i += 1
        self.all_targets = tuple(all_targets)
        self.reset_progress()
        self.load_components()

    def reset_progress(self):
        super().reset_progress()
        self.buildable = [b for b in self.all_targets if len(b.parents) == 0]

    def post_step(self):
        for i, supply in enumerate(self.supplies):
            if supply.cur_agent is None:
                agent = supply.buffer.dequeue()
                if agent is None: continue
                assert agent.allocated is not None
                agent.waiting = 0
                agent.pick_brick(supply, 10)
            elif supply.cur_agent.waiting == float('inf'):  # TODO bug: sometimes dequeued agent remain in waiting area and has to wait for inf before refill, don't know why
                supply.cur_agent.waiting = 0
            # else:
            #     supply.cur_agent.waiting = 0  # TODO delete!?

        for agent in self.agents:
            if agent.allocated is None and not agent.supervised:
                if len(self.buildable) and agent.battery > 35:
                    agent.allocated = self.buildable.pop(
                        np.random.randint(0, len(self.buildable)))
                    agent.assign_supply(self.supplies)
                    agent.target.state = copy.copy(agent.supply.state)
                else:
                    if agent.supervise_state != "charge":
                        agent.target.state = copy.copy(
                            self.docks[agent.uid].state)

        for agent in self.agents:
            if np.all(agent.state.p_pos == agent.dock.state.p_pos) and agent.state.p_alt == agent.dock.state.p_alt:
                continue
            agent.battery = max(1, agent.battery - 0.02)

        for agent in self.agents:
            d = np.linalg.norm(agent.state.p_pos - agent.target.state.p_pos)
            if not agent.supervised and d < agent.construct_range and (
                    np.abs(agent.state.p_vel) < agent.construct_maxvel).all():
                if agent.cur_construct_capacity >= 1:
                    # when agent reached target with brick
                    assert agent.allocated is not None
                    assert agent.load is not None
                    if agent.supervise_state != "build":
                        agent.lay_brick(self, 10)
                        print(agent.name, 'placed an element')
                        agent.supervise(agent.supervise_alt,
                                        [agent.flight_alt_unloaded])
                else:
                    if agent.supervise_state not in ["pick",
                                                     "queue"] and agent.allocated is not None:
                        if agent.supply.cur_agent is None and not len(agent.supply.buffer.queue):
                            agent.waiting = 0
                            agent.pick_brick(agent.supply, 10)
                        else:
                            agent.supervise_state = "queue"
                            pos = agent.supply.buffer.enqueue(agent)
                            agent.supervise(agent.supervise_pos, [pos])
                            agent.supervise(agent.supervise_alt,
                                            [agent.supply.buffer.h])
                            v = agent.supply.state.p_pos - pos
                            rot = v / np.linalg.norm(v)
                            agent.supervise(agent.supervise_rot, [
                                (-np.arctan2(*rot)) % (2 * np.pi)])
                            agent.supervise(setattr,
                                            [agent, 'waiting', float('inf')])
                            agent.supervise(agent.supervise_wait, [])

                    elif agent.supervise_state != "charge" and agent.allocated is None:
                        agent.supervise_state = "charge"
                        agent.supervise(agent.supervise_pos,
                                        [agent.target.state.p_pos])
                        agent.supervise(agent.supervise_rot,
                                        [agent.target.state.p_rot])
                        agent.supervise(agent.supervise_alt,
                                        [agent.target.state.p_alt])
                        agent.supervise(agent.supervise_charge, [self])
                        print(agent.name, 'start charging')

            if not agent.supervised and agent.supervise_state == "raise":  # TODO ?
                agent.supervise_state = None

        for agent in self.agents:
            assert len(agent.lidar_memory) == 2
            agent.lidar_memory.pop(0)
            agent.lidar_memory.append(agent.agents_lidar)
            agent.agents_lidar = self.lidar.get_ray_lidar(agent)

    def gather_stream_data(self):
        d = {'agents': [],
             'bricks': {'transport': [],
                        'built': []}}
        for i, a in enumerate(self.agents):
            d['agents'].append({'name': a.name,
                                'id': i,
                                'p_pos': np.around(a.state.p_pos, 3).tolist() + [round(a.state.p_alt, 4)],
                                'p_rot': round(a.state.p_rot, 3),
                                'battery': round(min(100, max(1, a.battery)), 3),
                                })
            if a.load:
                d['bricks']['transport'].append({
                    'p_pos': np.around(a.state.p_pos, 3).tolist() + [round(a.state.p_alt, 3)],
                    'p_rot': round(a.state.p_rot, 3),
                    'type': a.load.type,
                })
        for e in self.built:
            d['bricks']['built'].append({
                'p_pos': np.around(e.t_pos, 3).tolist() + [round(e.t_alt, 3)],
                'p_rot': round(e.t_rot, 3),
                'type': e.type,
            })
        return json.dumps(d)


class FacadeCoatingWorld(DroneConstructionWorld):

    def init_from_json(self, path):
        self.js_path = path
        with open(path, 'r') as f:
            self.js = json.load(f)
        all_targets = []
        sprays = self.js['sprays']
        for spray in sprays:
            assert spray['id'] == len(all_targets)
            s = SprayEntity(uid=spray['id'])
            s.state.p_pos = np.array(spray['pos'][:-1])
            s.state.p_rot = spray['rot']
            s.state.p_alt = spray['pos'][-1]
            s.type = None
            all_targets.append(s)
        self.all_targets = tuple(all_targets)
        self.reset_progress()
        self.load_components()

    def reset_progress(self):
        super().reset_progress()
        for spray in self.buildable:
            spray.progress = 0

    def post_step(self):
        for i, supply in enumerate(self.supplies):
            if supply.cur_agent is None:
                agent = supply.buffer.dequeue()
                if agent is None: continue
                assert agent.allocated is not None
                agent.waiting = 0
                agent.refill(supply, 50)
                assert agent.waiting == 0
            elif supply.cur_agent.waiting == float('inf'):  # TODO bug: sometimes dequeued agent remain in waiting area and has to wait for inf before refill, don't know why
                supply.cur_agent.waiting = 0

        for agent in self.agents:
            if agent.allocated is None and not agent.supervised:
                if len(self.buildable) and agent.battery > 35:
                    agent.allocated = self.buildable.pop(
                        np.random.randint(0, len(self.buildable)))
                    agent.assign_supply(self.supplies)
                    agent.target.state = copy.copy(agent.supply.state)
                else:
                    if agent.supervise_state != "charge":
                        agent.target.state = copy.copy(
                            self.docks[agent.uid].state)

        for agent in self.agents:
            if np.all(agent.state.p_pos == agent.dock.state.p_pos) and \
                    agent.state.p_alt == agent.dock.state.p_alt:
                continue
            agent.battery = max(1, agent.battery - 0.05)

        for agent in self.agents:
            d = np.linalg.norm(agent.state.p_pos - agent.target.state.p_pos)
            if not agent.supervised and d < agent.construct_range and (
                    np.abs(agent.state.p_vel) < agent.construct_maxvel).all():
                if agent.cur_construct_capacity >= 1:
                    # when agent reached target with material load
                    assert agent.allocated is not None
                    assert agent.load > 0
                    if agent.supervise_state != "build":
                        agent.spray(self)
                        print(agent.name, 'preparing to spray')
                        agent.supervise(agent.supervise_alt,
                                        [agent.flight_alt_unloaded])
                else:
                    if agent.supervise_state not in ["pick",
                                                     "queue"] and agent.allocated is not None:
                        if agent.supply.cur_agent is None and not len(agent.supply.buffer.queue):
                            agent.waiting = 0
                            agent.refill(agent.supply, 50)
                            assert agent.waiting == 0
                        else:
                            agent.supervise_state = "queue"
                            pos = agent.supply.buffer.enqueue(agent)
                            agent.supervise(agent.supervise_pos, [pos])
                            agent.supervise(agent.supervise_alt,
                                            [agent.supply.buffer.h])
                            v = agent.supply.state.p_pos - pos
                            rot = v / np.linalg.norm(v)
                            agent.supervise(agent.supervise_rot, [
                                (-np.arctan2(*rot)) % (2 * np.pi)])
                            agent.supervise(setattr,
                                            [agent, 'waiting', float('inf')])
                            agent.supervise(agent.supervise_wait, [])

                    elif agent.supervise_state != "charge" and agent.allocated is None:
                        agent.supervise_state = "charge"
                        agent.supervise(agent.supervise_pos,
                                        [agent.target.state.p_pos])
                        agent.supervise(agent.supervise_rot,
                                        [agent.target.state.p_rot])
                        agent.supervise(agent.supervise_alt,
                                        [agent.target.state.p_alt])
                        agent.supervise(agent.supervise_charge, [self])
                        print(agent.name, 'start charging')

            if not agent.supervised and agent.supervise_state == "raise":  # TODO ?
                agent.supervise_state = None

        for agent in self.agents:
            assert len(agent.lidar_memory) == 2
            agent.lidar_memory.pop(0)
            agent.lidar_memory.append(agent.agents_lidar)
            agent.agents_lidar = self.lidar.get_ray_lidar(agent)

    def gather_stream_data(self):
        d = {'agents': [],
             'sprays': {'progress': [s.progress for s in self.all_targets],
                        'operating': [s.spraying for s in self.all_targets]}}
        for i, a in enumerate(self.agents):
            d['agents'].append({'name': a.name,
                                'id': i,
                                'p_pos': np.around(a.state.p_pos, 3).tolist() + [round(a.state.p_alt, 4)],
                                'p_rot': round(a.state.p_rot, 3),
                                'battery': round(min(100, max(1, a.battery)), 3),
                                })
        return json.dumps(d)

