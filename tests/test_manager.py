def test_manager_works():
    from yucca.training.managers.YuccaManager import YuccaManager

    manager = YuccaManager()
    assert manager is not None
