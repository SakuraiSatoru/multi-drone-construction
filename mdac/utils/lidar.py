import numpy as np
from .entities import TargetLandmark, SupplyEntity, Drone


class AgentLidar(object):

    def __init__(self, env, n_lidar_per_agent=30):
        self.env = env
        self.n_lidar_per_agent = n_lidar_per_agent
        self.lidar_angles = np.linspace(0, 2*np.pi, num=self.n_lidar_per_agent, endpoint=False)

    @staticmethod
    def dist(entity1, entity2):
        return np.linalg.norm(entity1.state.p_pos - entity2.state.p_pos)

    @staticmethod
    def angle(entity1, entity2):
        if entity1 is entity2:
            raise Exception('measuring angle of itself')
        angle = np.arctan2(*np.flip(entity2.state.p_pos - entity1.state.p_pos)) + np.pi
        return angle + 0.5*np.pi if angle < 1.5*np.pi else angle - 1.5*np.pi

    def _get_lidar(self, agent, entities, lidar_range=False):
        lidars = dict()
        for entity in entities:
            d = AgentLidar.dist(agent, entity)
            if agent is entity or (lidar_range and d > agent.lidar_range):
                continue
            angle = AgentLidar.angle(agent, entity)
            section = int(angle // (2 * np.pi / self.n_lidar_per_agent))
            lidars[section] = min(d, lidars.get(section, float('inf')))
        return np.array(
            [lidars.get(i, 0.0) for i in range(self.n_lidar_per_agent)])

    def targets_lidar(self, agent):
        return self._get_lidar(agent, [i for i in self.env.landmarks if isinstance(i, TargetLandmark)])

    def supplies_lidar(self, agent):
        return self._get_lidar(agent, [i for i in self.env.landmarks if isinstance(i, SupplyEntity)])

    def agents_lidar(self, agent):
        return self._get_lidar(agent, self.env.agents, lidar_range=True)


class RayLidar(object):

    def __init__(self, env, n_lidar_per_agent):
        self.env = env
        self.ray_length = 2.0
        self.n_lidar_per_agent = n_lidar_per_agent
        self.lidar_angles = np.linspace(0, 2*np.pi, num=self.n_lidar_per_agent, endpoint=False)
        # counter-clockwise, start from right
        self.lidar_rays = np.array([np.cos(self.lidar_angles),
                                    np.sin(self.lidar_angles)]).T

    @staticmethod
    def dist(entity1, entity2):
        return np.linalg.norm(entity1.state.p_pos - entity2.state.p_pos)

    '''
    def _get_lidar(self, agent, entities):
        """
        Deprecated method.
        This method uses shapely library to find intersection, which is slower than directly computing circle & ray intersection.
        Note:
            1. current model is trained on this method
            2. this method is not exactly accurate, typically 1e-3 error
        """
        from shapely.geometry import Point, LineString
        rays = [LineString([agent.state.p_pos, agent.lidar_range*dir + agent.state.p_pos]) for dir in self.lidar_rays]
        lidar = np.full(self.n_lidar_per_agent, agent.lidar_range)
        for entity in entities:
            d = RayLidar.dist(agent, entity)
            if agent is entity or d >= agent.lidar_range + entity.size:
                continue
            if hasattr(agent.state, 'p_alt') and hasattr(entity.state, 'p_alt') and isinstance(entity, Drone):
                if agent.state.p_alt and entity.state.p_alt:
                    if abs(agent.state.p_alt - entity.state.p_alt) >= 0.1:
                        continue
            if d < agent.size:
                lidar.fill(0.0)
                return lidar
            c = Point(entity.state.p_pos).buffer(entity.size).boundary
            for i, ray in enumerate(rays):
                inter = c.intersection(ray)
                if hasattr(inter, 'geoms'):
                    pts = inter.geoms
                else:
                    assert isinstance(inter, Point)
                    pts = [inter]
                if not len(pts):
                    continue
                lidar[i] = min(lidar[i], *[np.linalg.norm(agent.state.p_pos - np.array(p.coords)) - agent.size for p in pts])
                # lidar[i] = min(lidar[i], *[np.linalg.norm(agent.state.p_pos - np.array(p.coords)) for p in pts])
        return lidar
    '''

    def _get_lidar(self, agent, entities):
        lidar = np.full(self.n_lidar_per_agent, agent.lidar_range)
        for entity in entities:
            if agent is entity:
                continue
            if hasattr(agent.state, 'p_alt') and hasattr(entity.state, 'p_alt') and isinstance(entity, Drone):
                if agent.state.p_alt and entity.state.p_alt:
                    if abs(agent.state.p_alt - entity.state.p_alt) >= 0.1:
                        continue
            pos_d = agent.state.p_pos - entity.state.p_pos
            for i, ray in enumerate(self.lidar_rays):
                b = np.dot(ray, pos_d) * 2
                d = b * b - 4 * (np.dot(pos_d, pos_d) - entity.size * entity.size)
                if d < 0:
                    continue
                d = np.sqrt(d)
                t = (-b - d) / 2
                if t > 0:
                    lidar[i] = max(min(lidar[i], t - agent.size), 0.0)
        return lidar

    def get_ray_lidar(self, agent):
        return self._get_lidar(agent, [a for a in self.env.agents if a.movable] + self.env.threats)
