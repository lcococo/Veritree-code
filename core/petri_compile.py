import contextlib
import io
from typing import Any

from .powl_json import TAU, spec_to_powl


def activities_of(spec: Any) -> set[str]:
    if isinstance(spec, str):
        return set() if spec == TAU else {spec}
    result = set()
    for child in spec["children"]:
        result.update(activities_of(child))
    return result


def spec_to_net(spec: Any):
    from pm4py import convert_to_petri_net

    return convert_to_petri_net(spec_to_powl(spec))


def net_labels(net: Any) -> set[str]:
    return {transition.label for transition in net.transitions if transition.label is not None}


def is_sound(net: Any, initial_marking: Any, final_marking: Any) -> bool:
    from pm4py.algo.analysis.woflan import algorithm as woflan

    with contextlib.redirect_stdout(io.StringIO()):
        return bool(woflan.apply(net, initial_marking, final_marking))


def check_instance(spec: Any) -> dict[str, Any]:
    net, initial_marking, final_marking = spec_to_net(spec)
    activities = activities_of(spec)
    labels = net_labels(net)
    return {
        "places": len(net.places),
        "transitions": len(net.transitions),
        "arcs": len(net.arcs),
        "activities": len(activities),
        "labels_match": labels == activities,
        "missing_labels": sorted(activities - labels),
        "extra_labels": sorted(labels - activities),
        "sound": is_sound(net, initial_marking, final_marking),
        "net": net,
        "initial_marking": initial_marking,
        "final_marking": final_marking,
    }
