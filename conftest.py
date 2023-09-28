from pytest import fixture

def pytest_addoption(parser):
    parser.addoption("--n_samples", type=int, action="store", default=1)


@fixture(scope='session')
def n_samples(request):
    return request.config.option.n_samples
    #return request.config.getoption("--n_samples")
