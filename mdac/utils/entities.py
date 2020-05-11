import copy
import numpy as np
from multiagent.core import Agent, Landmark, AgentState, Entity, EntityState


class Drone(Agent):

    def __init__(self, uid):
        super(Drone, self).__init__()
        self.uid = uid
        self.collide = True
        self.silent = True
        self.state.p_vel = np.zeros(2)
        self.state.p_rot = np.array([0., 1.])
        self.lidar_range = None
        self.agents_lidar = None
        self.pseudo_collision_range = None
        self.construct_range = None
        self.construct_time = None
        self.cur_construct_time = 0
        self.cur_construct_capacity = 0
        self.target = None  # TargetLandmark
        self.dock = None
        self.supply = None
        self.load = None  # brick
        self.allocated = None  # brick
        self.lidar_memory = []
        self.previous_state = AgentState()
        self.terminate = False
        self.flight_alt_loaded = 4.5
        self.flight_alt_unloaded = 4.5
        self.supervised = False
        self.supervise_fn = []
        self.supervise_param = []
        self.supervise_state = None
        self.battery = 100
        self.waiting = 0  # waiting time, float('inf') for infinite time
        self.act_hidden = None
        self.crt_hidden = None

    def supervise(self, fn, param):
        self.supervised = True
        self.supervise_fn.append(fn)
        self.supervise_param.append(param)
        self.state.p_vel = np.zeros(2)

    def supervise_pos(self, p_pos):
        d = p_pos - self.state.p_pos
        n = np.linalg.norm(d)
        if n > 0:
            rot = d / n
            self.state.p_rot = (-np.arctan2(*rot)) % (2 * np.pi)
        else:
            self.state.p_rot = 0
        if np.any(np.abs(d) > 0.05):
            self.state.p_pos += 0.08 * d / np.linalg.norm(d)
            return False
        else:
            self.state.p_pos = np.copy(p_pos)
            return True

    def supervise_rot(self, p_rot):
        d = abs(self.state.p_rot - p_rot)
        if min(d, 2 * np.pi - d) > 0.1:
            self.state.p_rot += 0.15 * ((self.state.p_rot < p_rot) * 2 - 1) * (
                    (d <= np.pi) * 2 - 1)
            return False
        else:
            self.state.p_rot = p_rot
            return True

    def supervise_alt(self, p_alt):
        d = self.state.p_alt - p_alt
        if abs(d) > 0.1:
            self.state.p_alt += 0.15 * ((d < 0) * 2 - 1)
            return False
        else:
            self.state.p_alt = p_alt
            return True

    def supervise_refill(self, spray, amount=0.5):
        if self.load is None: self.load = 0
        self.load += amount
        self.target.state = copy.copy(spray.state)
        self.construct_range = self.size * 1.5
        self.cur_construct_capacity = 1
        print('%s refilled %f material' % (self.name, amount))

    def supervise_pick(self, brick):
        self.load = brick
        self.target.state.p_pos = brick.t_pos
        self.target.state.p_rot = brick.t_rot
        self.target.state.p_alt = brick.t_alt
        self.cur_construct_capacity = 1
        self.construct_range = self.size * 1.5
        print('%s picked up an element' % self.name)

    def supervise_lay_brick(self, world):
        brick = self.load
        brick.state.p_pos = np.copy(brick.t_pos)
        brick.state.p_rot = brick.t_rot
        brick.state.p_alt = brick.t_alt
        brick.color = self.color
        brick.size = 0.4
        world.built.append(brick)
        self.cur_construct_capacity = 0
        self.construct_range = world.supplies[0].size - self.size
        world.buildable += brick.notify_built()
        self.allocated = None
        self.load = None

    def supervise_spray(self, world):
        self.allocated.progress = min(1, self.allocated.progress + self.load)
        self.load = 0
        self.cur_construct_capacity = 0
        if self.allocated.progress < 1:
            world.buildable.append(self.allocated)
        self.allocated = None

    def supervise_charge(self, world):
        if self.allocated:
            raise NotImplementedError
        # TODO battery remain
        if len(world.buildable) and self.battery >= 100:
            self.allocated = world.buildable.pop(
                np.random.randint(0, len(world.buildable)))
            self.assign_supply(world.supplies)
            self.target.state = copy.copy(self.supply.state)
            # self.target.state = copy.copy(world.supplies[0].state)
            self.supervise(self.supervise_alt, [self.flight_alt_unloaded])
            return True
        self.battery = min(100, 0.1 + self.battery)
        return False

    def supervise_wait(self):
        if self.waiting == float('inf'):
            return False
        self.waiting -= 1
        return self.waiting <= 0

    def pick_brick(self, supply, time=10):
        self.supervise_state = "pick"
        supply.cur_agent = self
        self.supervise(self.supervise_pos, [self.target.state.p_pos])
        self.supervise(self.supervise_rot, [self.target.state.p_rot])
        self.supervise(self.supervise_alt, [self.target.state.p_alt])
        self.supervise(setattr, [self, 'waiting', time])
        self.supervise(self.supervise_wait, [])
        self.supervise(self.supervise_pick, [self.allocated])
        self.supervise(self.supervise_alt, [self.flight_alt_loaded])
        self.supervise(setattr, [supply, 'cur_agent', None])

    def lay_brick(self, world, time=10):
        self.supervise_state = "build"
        self.supervise(self.supervise_pos,
                        [self.load.t_pos])
        self.supervise(self.supervise_rot,
                        [self.load.t_rot])
        self.supervise(self.supervise_alt,
                        [self.load.t_alt])
        self.supervise(setattr, [self, 'waiting', time])
        self.supervise(self.supervise_wait, [])
        self.supervise(self.supervise_lay_brick, [world])

    def refill(self, supply, time):
        self.supervise_state = "pick"
        supply.cur_agent = self
        self.supervise(self.supervise_pos, [self.target.state.p_pos])
        self.supervise(self.supervise_rot, [self.target.state.p_rot])
        self.supervise(self.supervise_alt, [self.target.state.p_alt])
        self.supervise(setattr, [self, 'waiting', time])
        self.supervise(self.supervise_wait, [])
        self.supervise(self.supervise_refill, [self.allocated])
        self.supervise(self.supervise_alt, [self.flight_alt_loaded])
        self.supervise(setattr, [supply, 'cur_agent', None])

    def spray(self, world):
        self.supervise_state = "build"
        self.construct_range = world.supplies[0].size - self.size
        self.supervise(self.supervise_pos,
                       [self.target.state.p_pos])
        self.supervise(self.supervise_rot,
                       [self.target.state.p_rot])
        self.supervise(self.supervise_alt,
                       [self.target.state.p_alt])
        self.supervise(setattr, [self.allocated, 'spraying', True])
        self.supervise(setattr, [self, 'waiting', 100])
        self.supervise(self.supervise_wait, [])
        self.supervise(self.supervise_spray, [world])
        self.supervise(setattr, [self.allocated, 'spraying', False])

    def assign_supply(self, supplies):
        assert self.allocated is not None
        brick_type = self.allocated.type
        self.supply = None
        min_d = float('inf')
        for s in supplies:
            if brick_type is not None and s.type != brick_type: continue
            if self.supply is None:
                self.supply = s
                min_d = np.linalg.norm(self.state.p_pos - s.state.p_pos)
                continue
            if len(s.buffer.queue) < len(self.supply.buffer.queue):
                self.supply = s
            elif len(s.buffer.queue) == len(self.supply.buffer.queue):
                if min_d > np.linalg.norm(self.state.p_pos - s.state.p_pos):
                    self.supply = s


class TargetLandmark(Landmark):

    def __init__(self):
        super(TargetLandmark, self).__init__()
        self.collide = False
        self.movable = False
        self.state.p_vel = np.zeros(2)


class DroneWorldEntity(Entity):

    def __init__(self, uid):
        super().__init__()
        self.collide = False
        self.movable = False
        self.uid = uid
        self.state.p_pos = np.zeros(2)
        self.state.p_vel = np.zeros(2)
        self.state.p_rot = 0
        self.state.p_alt = 0
        self.name = "%s %d" % (type(self), uid)


class SupplyEntity(DroneWorldEntity):

    def __init__(self, uid, type=None):
        super(SupplyEntity, self).__init__(uid)
        self.color = np.array([0, 0.85, 0])
        self.type = type
        self.cur_agent = None
        self.buffer = None


class DockEntity(DroneWorldEntity):

    def __init__(self, uid):
        super(DockEntity, self).__init__(uid)


class ThreatEntity(DroneWorldEntity):

    def __init__(self, uid):
        super(ThreatEntity, self).__init__(uid)
        self.collide = True
        self.color = np.array([0.25, 0.25, 0.25])


class BrickEntity(DroneWorldEntity):

    def __init__(self, uid, target_pos, target_rot, parents=None, type=None):
        super().__init__(uid)
        self.t_pos = target_pos[:2]
        self.t_alt = target_pos[2]
        self.t_rot = target_rot
        self.type = type
        self.parents = set(parents) if parents else set()
        self.childen = set()
        for parent in self.parents:
            parent.register_child(self)

    def register_parent(self, brick):
        self.parents.add(brick)
        brick.register_child(self)

    def notify_built(self):
        buildable = []
        for child in self.childen:
            child.parents.remove(self)
            if len(child.parents) == 0:
                buildable.append(child)
        return buildable

    def register_child(self, brick):
        self.childen.add(brick)


class SprayEntity(DroneWorldEntity):

    def __init__(self, uid, build_speed = 0.2, type=None):
        super().__init__(uid)
        self.collide = False
        self.movable = False
        self.type = type
        self.state = EntityState()
        self.state.p_vel = np.zeros(2)
        self.progress = 0
        self.build_speed = build_speed
        self.spraying = False


class SupplyBuffer(DroneWorldEntity):

    def __init__(self, uid, supply_entity, r=1.5, n=8, h=2):
        super(SupplyBuffer, self).__init__(uid)
        supply_entity.buffer = self
        self.c = supply_entity.state.p_pos
        self.r = r
        self.n = n
        self.h = h

        angles = np.linspace(0.5 * np.pi, 2.5 * np.pi, num=self.n,
                             endpoint=False)
        # counter-clockwise, start from right
        anchors = list(
            np.array([np.cos(angles), np.sin(angles)]).T * self.r + self.c)
        self.anchors = []
        p = 0
        while len(anchors):
            self.anchors.append(anchors.pop(p))
            p = - p - 1
        self.queue = []  # [[anchor_index, agent], ...]

    def enqueue(self, agent):
        if len(self.queue) >= self.n:
            raise NotImplementedError
        indexes = list(range(self.n))
        for i, _ in self.queue:
            indexes.remove(i)
        i = np.argmin(
            [np.linalg.norm(agent.state.p_pos - self.anchors[i]) for i in
             indexes])
        self.queue.append([indexes[i], agent])
        return self.anchors[indexes[i]]

    def dequeue(self):
        if len(self.queue):
            i, agent = self.queue.pop(0)
            return agent
        return None
